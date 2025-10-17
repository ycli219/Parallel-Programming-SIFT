#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <chrono>

#include "image.hpp"
#include "sift.hpp"
#include <mpi.h>

namespace {
void broadcast_image(Image& img, int root, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    int meta[3];
    if (rank == root) {
        meta[0] = img.width;
        meta[1] = img.height;
        meta[2] = img.channels;
    }
    MPI_Bcast(meta, 3, MPI_INT, root, comm);
    if (rank != root) {
        img = Image(meta[0], meta[1], meta[2]);
    }
    int total = meta[0] * meta[1] * meta[2];
    if (total > 0) {
        MPI_Bcast(img.data, total, MPI_FLOAT, root, comm);
    }
}
}

int main(int argc, char *argv[])
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    if (argc != 4) {
        std::cerr << "Usage: ./hw2 ./testcases/xx.jpg ./results/xx.jpg ./results/xx.txt\n";
        return 1;
    }

    MPI_Init(&argc, &argv);
    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc != 4) {
        if (world_rank == 0) {
            std::cerr << "Usage: ./hw2 ./testcases/xx.jpg ./results/xx.jpg ./results/xx.txt\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::string input_img = argv[1];
    std::string output_img = argv[2];
    std::string output_txt = argv[3];
    
    Image img;
    if (world_rank == 0) {
        img = Image(input_img);
        if (img.channels != 1) {
            img = rgb_to_grayscale(img);
        }
    }
    broadcast_image(img, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    std::chrono::high_resolution_clock::time_point start;
    if (world_rank == 0) {
        start = std::chrono::high_resolution_clock::now();
    }

    std::vector<Keypoint> kps = find_keypoints_and_descriptors_parallel(img, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Execution time: " << duration.count() << " ms\n";
        std::cout << "Found " << kps.size() << " keypoints.\n";

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
    }
    
    MPI_Finalize();
    return 0;
}