#pragma once

void Conv3d_16b_ser(SHORT *psh_3D_sourse, const int i_WH, const int i_NinStack, 
					SHORT *psh_3D_dest, int *piKernel27, const int iNormShift);	

bool Conv3d_16b_SSE(SHORT *psh_3D_sourse, const int i_WH, const int i_NinStack, 
					SHORT *psh_3D_dest, int *piKernel27, const int iNormShift,
					int *pi_WorkingLines);	

bool Conv2d_16to32_1line_SSE(SHORT *pshLine_in_Center, const int i_WH, int *piKernel9, int *piLineOut);


bool Grad3d_16b_SSE(SHORT *psh_3D_sourse, const int i_WH, const int i_NinStack, 
					USHORT *psh_Grad3D_dest, 
					int *piKernelX27, int *piKernelY27, int *piKernelZ27, 
					float flt_XY_scale,
					const int iNormShift,
					int *pi_WorkingLines);	


int Diff3d_16b(SHORT *psh_3D_1, const int i_WH, const int i_NinStack, SHORT *psh_3D_2);