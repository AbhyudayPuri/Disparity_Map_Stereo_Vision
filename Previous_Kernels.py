		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		__syncthreads();
		printf("blocksize %d", block_size[0]);
		printf("row: %d \\n", row);
		printf("block id: %d", blockIdx.y);
		int disparity = 0; 
		int max_cost = 65536;
		if(row < L_height && col < L_width){
				for(int k = 0; k < 50; k++){
					__syncthreads();
					printf("HI");
					//if((row * block_size[0] - 25 + k) >= 0 && (row * block_size[0] - 25 + k) < (L_height - block_size[0]) && (col * block_size[1] - 25 + k) >= 0 && (col * block_size[1] - 25 + k) < (L_width - block_size[1])){
						__syncthreads();
						printf("Hello there");
						int cost = 0;
						for(int m = 0; m < block_size[0]; m++){
							for(int n = 0; n < block_size[1]; n++){
								cost += (L[(row * block_size[0] +m)* L_width + (col * block_size[1] + n)]); //- R[(row * block_size[0] - 25 + k + m)*L_width + col * block_size[0] - 25 + k + n]) * (L[(row * block_size[0] +m)* L_width + (col * block_size[1] + n)] - R[(row * block_size[0] - 25 + k + m)*L_width + col * block_size[0] - 25 + k + n]);
								// cost += (L[(row * block_size[0] + m) * L_width + (col * block_size[1] + n)] - R[(row * block_size[0] + m)*L_width + col * block_size[0] + n]) * (L[(row * block_size[0] + m) * L_width + (col * block_size[1] + n)] - R[(row * block_size[0] + m)*L_width + col * block_size[0] + n]);  

							}
						}
						__syncthreads();
						if(cost < max_cost){
							max_cost = cost;
							disparity = abs(25 - k);
						}
					//}
				}
			//for(int m = 0; m < block_size[0]; m++){
				//for(int n = 0; n < block_size[1]; n++){
					//D[(row * block_size[0] + m)*L_width+ col * block_size[1] + n] = disparity;
				//}
			//}	
		}
#################################################################################################################################################################################################################################################################################################################################################################
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;

		 if((row < L_height) && (col < L_width))
		 {
		 	int disparity = 128;
		 	int max_cost = 999999;

		 	for(int k = 0; k < 50; k++)
		 	{	
		 		int cost = 0;
		 		for(int m = 0; m < block_size[0]; m++)
		 		{
		 			for(int n = 0; n < block_size[1]; n++)
		 			{
		 				cost += (L[(row + m) * L_width + col + n] - R[(row + m) * L_width + col + n]) * (L[(row + m) * L_width + col + n] - R[(row + m) * L_width + col + n]) ;
		 			}
		 		}
		 		if(cost < max_cost)
		 		{
		 			max_cost = cost;
		 			disparity = k;
		 		}
		 	}
		 	for(int m = 0; m < block_size[0]; m++)
		 	{
		 		for(int n = 0; n < block_size[1]; n++)
		 		{
		 			D[(row + m) * L_width + col + n] = disparity;
		 		}
		 	}
		 }
#################################################################################################################################################################################################################################################################################################################################################################




