#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <chrono>
#include <mpi.h>
#include <omp.h>

#include "image.hpp"
#include "sift.hpp"


int main(int argc, char *argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    if (argc != 4) {
        if (rank == 0) {
            std::cerr << "Usage: ./hw2 ./testcases/xx.jpg ./results/xx.jpg ./results/xx.txt\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::string input_img = argv[1];
    std::string output_img = argv[2];
    std::string output_txt = argv[3];

    omp_set_num_threads(6);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Only rank 0 loads the image initially
    Image img;
    if (rank == 0) {
        img = Image(input_img);
        img = img.channels == 1 ? img : rgb_to_grayscale(img);
    }

    // Broadcast image dimensions to all processes
    int width, height;
    if (rank == 0) {
        width = img.width;
        height = img.height;
    }
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Create image on other processes
    if (rank != 0) {
        img = Image(width, height, 1);
    }

    // Broadcast image data to all processes
    MPI_Bcast(img.data, width * height, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // All processes compute SIFT features
    std::vector<Keypoint> kps = find_keypoints_and_descriptors(img);

    if (rank == 0) {
        /////////////////////////////////////////////////////////////
        // The following code is for the validation
        // You can not change the logic of the following code, because it is used for judge system
        std::ofstream ofs(output_txt);
        if (!ofs) {
            std::cerr << "Failed to open " << output_txt << " for writing.\n";
        } else {
            ofs << kps.size() << "\n";
            for (const auto& kp : kps) {
                ofs << kp.i << " " << kp.j << " " << kp.octave << " " << kp.scale << " ";
                for (size_t i = 0; i < kp.descriptor.size(); ++i) {
                    ofs << " " << static_cast<int>(kp.descriptor[i]);
                }
                ofs << "\n";
            }
            ofs.close();
        }

        Image result = draw_keypoints(img, kps);
        result.save(output_img);
        /////////////////////////////////////////////////////////////

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Execution time: " << duration.count() << " ms\n";
        std::cout << "Found " << kps.size() << " keypoints.\n";
    }

    MPI_Finalize();
    return 0;
}