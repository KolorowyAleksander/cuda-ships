#include <string>
#include <vector>
#include <list>
#include <tuple>

#include <algorithm>

#include <fstream>
#include <sstream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>

#include "check_cuda_error.h"

#define BLOCK_DIM_2D 32
#define BLOCK_DIM 1024
#define GRID_DIM 4
#define WARP_SIZE 4
#define CHUNK_SIZE 8


typedef std::tuple <
	// std::vector<int>, // v
	// std::vector<int>, // e
	// std::vector<int>, // bfs
	// std::vector<int>, // colors
	thrust::device_vector<int>, // v
	thrust::device_vector<int>, // e
	thrust::device_vector<int>, // bfs
	thrust::device_vector<int>, // colors
	size_t				  // cols
> graph_data;

typedef std::tuple <
	std::vector<int>, // map
	size_t,              // cols
	size_t               // rows
> map_data;


map_data read_from_file(std::string const& filename) {
	std::ifstream file;

	file.open(filename, std::ios::in);

	if (file.good()) {
		size_t prev_cols_count = INT_MAX, cols_count = 0;
		size_t rows_count = 0;
		std::vector<int> map;

		for (std::string line; getline(file, line);) {
			cols_count = 0;
			std::istringstream is(line);

			for (int number; is >> number;){
				cols_count += 1;
				map.push_back(number);
			}

			if ((prev_cols_count != INT_MAX) && (cols_count != prev_cols_count)) {
				std::cerr << "Input file malformed: " << filename << std::endl;
				throw std::runtime_error("bad data format in file: " + filename);
			}

			prev_cols_count = cols_count;
			rows_count += 1;
		}

		return std::make_tuple(
			map,
			cols_count,
			rows_count
		);
	}
	else {
		std::cerr << "Could not read file: " << filename << std::endl;
		throw std::runtime_error("error reading file: " + filename);
	}
}

void save_to_file(const std::string &filename, std::vector<int> const& colors, size_t cols) {
	std::ofstream file;

	file.open(filename, std::ios::out);

	if (file.good()) {
		for (size_t i = 0; i < colors.size(); ++i) {
			file << (colors[i] >= 0 ? colors[i] : 0);
			if ((i + 1) % cols == 0) {
				file << std::endl;
			}
			else {
				file << " ";
			}
		}

		file.close();

	}
	else {
		std::cerr << "Cannot open file to save: " << filename << std::endl;
		throw std::runtime_error("bad file");
	}
}

// change 2d array indexes to 1d array index
int to_index(int x, int y, int cols) {
	return x * cols + y;
}

// checks if there are equal negative cells in &colors first and last row
std::vector<int> check_if_route_exists(std::vector<int> const& colors, size_t cols, size_t rows) {
	std::list<int> first_row, last_row;

	for (size_t i = 0; i < rows; i++) {
		int first_row_element = colors[to_index(i, 0, cols)];
		int last_row_element = colors[to_index(i, cols - 1, cols)];

		if (first_row_element < -1) {
			first_row.push_back(first_row_element);
		}

		if (last_row_element < -1) {
			last_row.push_back(last_row_element);
		}
	}

	first_row.sort();
	last_row.sort();

	std::vector<int> out;
	std::set_intersection(
		first_row.begin(),
		first_row.end(),
		last_row.begin(),
		last_row.end(),
		std::back_inserter(out)
	);

	return out;
}

// finds the route between the first and the last row and saves it to &colors
// requires the route to exist, if it doesn't, this will loop
void find_route(
	std::vector<int> &colors,    // colors
	std::vector<int> const& bfs, // bfs
	std::vector<int> e,          // vector of which paths connect both ends
	size_t cols,
	size_t rows
) {
	int position = 0;

	for (size_t i = 0; i < rows; i++) {
		auto it = std::find(e.begin(), e.end(), colors[to_index(i, 0, cols)]);
		if (it != e.end()) {
			position = i;
			break;
		}
	}

	size_t x = position, y = 0;
	while (y < cols - 1) {
		int current_bfs = bfs[to_index(x, y, cols)];

		colors[to_index(x, y, cols)] = 1;
		if (bfs[to_index(x, y + 1, cols)] == current_bfs + 1) {
			y += 1;
		} else if (x > 0 && bfs[to_index(x - 1, y, cols)] == current_bfs + 1) {
			x -= 1;
		} else if (x < (rows - 1) && bfs[to_index(x + 1 , y, cols)] == current_bfs + 1) {
			x += 1;
		} else if (y >= 1 && bfs[to_index(x, y - 1, cols)] == current_bfs + 1) {
			y -= 1;
		}
	}

	colors[to_index(x, y, cols)] = 1;
}

// each thread counts how many neighbours (edges) it's vertice in *M has
__global__ void neighbours_count_kernel(
	int *M,
	int *VC,
	int *C,
	int cols,
	int rows
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= cols || y >= rows) {
		return;
	}

	int index = x + y * cols;

	if (M[index] == 1) {
		// i = x - 1, j = y - 1;
		// i = x + 1, j = y - 1;
		// i = x - 1, j = y + 1;
		// i = x + 1, j = y + 1;
		// i = x, j = y - 1;
		// i = x - 1, j = y;
		// i = x + 1, j = y;
		// i = x, j = y + 1;

		int vertice_count = 0;

		for (int i = (x - 1); i <= (x + 1); i += 1) {
			for (int j = (y - 1); j <= (y + 1); j += 1) {
				if (!(i < 0 || i >= cols || j < 0 || j >= rows) && !(i == x && j == y)) {
					int index2 = i + j * cols;

					if (M[index2] == 1) {
						vertice_count += 1;
					}
				}
			}
		}

		VC[index + 1] = vertice_count;
		C[index] = 1;
	} else {
		bool next_to_1 = false;
		int vertice_count = 0;
		// i = x - 1, j = y - 1;
		// i = x + 1, j = y - 1;
		// i = x - 1, j = y + 1;
		// i = x + 1, j = y + 1;

		// diagonal neighbours indexes
		for (int i = (x - 1); i <= (x + 1); i += 2) {
			for (int j = (y - 1); j <= (y + 1); j += 2) {
				if (!(i < 0 || i >= cols || j < 0 || j >= rows)) {
					int index2 = i + j * cols;

					if (M[index2] == 1 && M[index] == 0) {
						next_to_1 = true;
					}
				}
			}
		}

		// i = x, j = y - 1
		// i = x - 1, j = y
		// i = x + 1, j = y
		// i = x, j = y + 1
		int i = x;
		for (int j = y - 1; j <= y + 1; j += 2) {
			int index2 = i + j * cols;

			if (!(i < 0 || i >= cols || j < 0 || j >= rows)) {
				if (M[index2] == 0) {
					vertice_count += 1;
				}

				if (M[index2] == 1) {
					next_to_1 = true;
				}
			}
		}

		int j = y;
		for (int i = (x - 1); i <= (x + 1); i += 2) {
			int index2 = i + j * cols;

			if (!(i < 0 || i >= cols || j < 0 || j >= rows)) {
				if (M[index2] == 0) {
					vertice_count += 1;
				}

				if (M[index2] == 1) {
					next_to_1 = true;
				}
			}
		}

		if (next_to_1 == false) {
			VC[index + 1] = vertice_count;
			C[index] = 0;
		} else {
			C[index] = -1;
		}
	}
}

// takes the 1d array M and creates V and E
__global__ void edges_creation_kernel(
	int *M,
	int *V,
	int *E,
	int cols,
	int rows
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= cols || y >= rows) {
		return;
	}

	int index = x + y * cols;

	if (M[index] == 1) {
		// i = x - 1, j = y - 1;
		// i = x + 1, j = y - 1;
		// i = x - 1, j = y + 1;
		// i = x + 1, j = y + 1;
		// i = x, j = y - 1;
		// i = x - 1, j = y;
		// i = x + 1, j = y;
		// i = x, j = y + 1;

		int vertice_count = 0;

		for (int i = (x - 1); i <= (x + 1); i += 1) {
			for (int j = (y - 1); j <= (y + 1); j += 1) {
				if (!(i < 0 || i >= cols || j < 0 || j >= rows) && !(i == x && j == y)) {
					int index2 = i + j * cols;

					if (M[index2] == 1) {
						int e_index = V[index];

						E[e_index + vertice_count] = index2;

						vertice_count += 1;
					}
				}
			}
		}
	} else {
		bool next_to_1 = false;
		int vertice_count = 0;
		// i = x - 1, j = y - 1;
		// i = x + 1, j = y - 1;
		// i = x - 1, j = y + 1;
		// i = x + 1, j = y + 1;
		// i = x, j = y - 1;
		// i = x - 1, j = y;
		// i = x + 1, j = y;
		// i = x, j = y + 1;

		// diagonal neighbours indexes
		for (int i = (x - 1); i <= (x + 1); i += 1) {
			for (int j = (y - 1); j <= (y + 1); j += 1) {
				if (!(i < 0 || i >= cols || j < 0 || j >= rows) && !(i == x && j == y)) {
					int index2 = i + j * cols;

					if (M[index2] == 1 && M[index] == 0) {
						next_to_1 = true;
					}
				}
			}
		}

		// i = x, j = y - 1
		// i = x - 1, j = y
		// i = x + 1, j = y
		// i = x, j = y + 1
		int i = x;
		for (int j = y - 1; j <= y + 1; j += 2) {
			int index2 = i + j * cols;

			if (!(i < 0 || i >= cols || j < 0 || j >= rows)) {
				if (M[index2] == 0 && next_to_1 == false) {
					int e_index = V[index];

					E[e_index + vertice_count] = index2;

					vertice_count += 1;
				}

			}
		}

		int j = y;
		for (int i = (x - 1); i <= (x + 1); i += 2) {
			int index2 = i + j * cols;

			if (!(i < 0 || i >= cols || j < 0 || j >= rows)) {
				if (M[index2] == 0 && next_to_1 == false) {
					int e_index = V[index];

					E[e_index + vertice_count] = index2;

					vertice_count += 1;
				}

			}
		}
	}
}

// runs kernels which fill e, v and bfs in the graph data
graph_data create_graph_data(map_data &md) {
	std::vector<int> const& map = std::get<0>(md);
	size_t cols = std::get<1>(md);
	size_t rows = std::get<2>(md);

	dim3 dimBlock(BLOCK_DIM_2D, BLOCK_DIM_2D);
	dim3 dimGrid((cols / dimBlock.x) + 1, (rows / dimBlock.y) + 1);

	thrust::device_vector<int> map_device(map);
	thrust::device_vector<int> vertices_device((rows * cols) + 1);
	thrust::device_vector<int> colors_device(rows * cols, 0);
	thrust::device_vector<int> bfs_device(rows * cols, INT_MAX);

	// find number of neighbours for each vertice
	neighbours_count_kernel <<<dimGrid, dimBlock>>> (
		map_device.data().get(),
		vertices_device.data().get(),
		colors_device.data().get(),
		(int)cols,
		(int)rows
	);
	CudaCheckError();

	// count v for bfs
	thrust::inclusive_scan(vertices_device.begin(), vertices_device.end(), vertices_device.begin());

	// fill e
	thrust::device_vector<int> edges_device(vertices_device.back());
	edges_creation_kernel <<<dimGrid, dimBlock >>> (
		map_device.data().get(),
		vertices_device.data().get(),
		edges_device.data().get(),
		(int)cols,
		(int)rows
	);
	CudaCheckError();

	return std::make_tuple(
		vertices_device,
		edges_device,
		bfs_device,
		colors_device,
		cols
	);
}



// WARPSIZE – liczba w¹tków w warpie,
// CHUNKSIZE – liczba wêz³ów przetwarzanych przez warp
// INFINITY – liczba okreœlaj¹ca wartoœæ BFS wêz³ów jeszcze nie odwiedzonych
template<int WARPSIZE, int CHUNKSIZE, int INF>
__global__ void bfs_kernel(
	const int *V,
	int *E,
	int *BFS,
	int *C,
	int color,
	unsigned long N,
	int curr,
	bool *finished
) {
    extern __shared__ int base[];
    int t = blockIdx.x * blockDim.x + threadIdx.x; //globalny nr w¹tku
    int lane = threadIdx.x % WARPSIZE;             //nr w¹tku w warpie
    int warpLocal = threadIdx.x / WARPSIZE;        //nr warpa w bloku
    int warpGlobal = t / WARPSIZE;                 //globalny nr warpa
    int warpLocalCount = blockDim.x / WARPSIZE;    //liczba warpow w bloku

	//Równoleg³e przepisanie czêœci danych do pamiêci wspó³dzielonej
	//(pêtla sekwencyjno-równoleg³a)
    int *sV = base + warpLocal * (CHUNKSIZE + 1);
    int *sBFS = base + warpLocalCount * (CHUNKSIZE + 1) + warpLocal * CHUNKSIZE;
    for (int i = lane; i < CHUNKSIZE + 1; i += WARPSIZE) { //Przepisz jedn¹ wartoœæ wiêcej
        if (warpGlobal * CHUNKSIZE + i < N + 1) {
            sV[i] = V[warpGlobal * CHUNKSIZE + i];
        }
    }
    for (int i = lane; i < CHUNKSIZE; i += WARPSIZE) {
        if (warpGlobal * CHUNKSIZE + i < N) {
            sBFS[i] = BFS[warpGlobal * CHUNKSIZE + i];
        }
    }

    __threadfence_block(); //Wszystkie w¹tki powinny widzieæ odczytane dane
    for (int v = 0; v < CHUNKSIZE; v++) {    //Przegl¹daj kolejne wêz³y w zbiorze warpa
        if (v + warpGlobal * CHUNKSIZE < N) {  //Je¿eli nie wychodzimy poza tablicê
            if (sBFS[v] == curr) {   //Je¿eli do wêz³a dotarto w poprzedniej iteracji
				//Iteruj po s¹siadach
                int num_nbr = sV[v + 1] - sV[v]; //Liczba s¹siadów (po to 1 element wiêcej)
                int *nbrs = &E[sV[v]];           //WskaŸnik na listê s¹siadów
				//”Równoleg³o – sekwencyjna” pêtla przegl¹daj¹ca s¹siadów wêz³a v
                for (int i = lane; i < num_nbr; i += WARPSIZE) {
                    int w = nbrs[i];             //Numer s¹siada
                    if (BFS[w] == INF && C[w] != -1) {         //Je¿eli s¹siada jeszcze nie odwiedzono
                        *finished = false;       //Konieczna ponowna iteracja
                        BFS[w] = curr + 1;       //Zapisz numer BFS s¹siada
                        
                        C[w] = color;
                    }
                }
                __threadfence_block();       //Wszystkie w¹tki powinny zobaczyæ zapisy
            }
        }
    }
}

void run_bfs(
	graph_data &gd,
	int start,
	int color
) {
	thrust::device_vector<int> &vertices = std::get<0>(gd);
	thrust::device_vector<int> &edges = std::get<1>(gd);
	thrust::device_vector<int> &bfs = std::get<2>(gd);
	thrust::device_vector<int> &colors = std::get<3>(gd);
	size_t cols = std::get<4>(gd);

	size_t vertices_number = vertices.size() - 1;

	bfs[start] = 0;
	colors[start] = color;

	int shared_mem_size = 2 * BLOCK_DIM / WARP_SIZE * (CHUNK_SIZE + 1) * sizeof(int);
	
	// variable for checking if the algorithm finished
	bool finished = true, *finished_device;
	CudaSafeCall(cudaMalloc((void **)&finished_device, sizeof(bool)));

	// perform computation actions
	unsigned int curr = 0;
	do {
		finished = true;
		CudaSafeCall(cudaMemcpy(finished_device, &finished, sizeof(bool), cudaMemcpyHostToDevice));

		bfs_kernel<WARP_SIZE, CHUNK_SIZE, INT_MAX>
			<<<GRID_DIM, BLOCK_DIM, shared_mem_size>>> (
			vertices.data().get(),
			edges.data().get(),
			bfs.data().get(),
			colors.data().get(),
			color,
			vertices_number,
			curr,
			finished_device
		);
		CudaCheckError();

		CudaSafeCall(cudaMemcpy(&finished, finished_device, sizeof(bool), cudaMemcpyDeviceToHost));
		curr++;
	} while (!finished && curr <= vertices_number);
}


int main(int argc, char **argv) {
    if (argc < 2) {
		std::cerr << "No input file specified." << std::endl;
        return EXIT_FAILURE;
    }

	map_data md = read_from_file(argv[1]);
	size_t cols = std::get<1>(md);
	size_t rows = std::get<2>(md);

	graph_data gd = create_graph_data(md);
	thrust::device_vector<int> &bfs = std::get<2>(gd);
	thrust::device_vector<int> &colors = std::get<3>(gd);
	
	// find islands
    int color = 2;
    for (unsigned long i = 0; i < rows * cols; ++i) {
        if (bfs[i] == INT_MAX && colors[i] != 0 && colors[i] != -1) {
            run_bfs(gd, i, color);
            color += 1;
        }
    }

	// find routes from left to right
	color = -2;
	for (unsigned long i = 0; i < rows * cols; i += cols) {
		if (bfs[i] == INT_MAX && (colors[i] == 0)) {
			run_bfs(gd, i, color);
			color -= 1;
		}
	}

	std::vector<int> colors_h(rows * cols);
	thrust::copy(colors.begin(), colors.end(), colors_h.begin());

	std::vector<int> s = check_if_route_exists(colors_h, cols, rows);
    if (s.size() != 0) {
		std::vector<int> bfs_h(rows * cols);
		thrust::copy(bfs.begin(), bfs.end(), bfs_h.begin());

		find_route(colors_h, bfs_h, s, cols, rows);
        save_to_file("out.txt", colors_h, cols);
    } else {
        std::cerr << "No route found." << std::endl;
    }

    return EXIT_SUCCESS;
}
