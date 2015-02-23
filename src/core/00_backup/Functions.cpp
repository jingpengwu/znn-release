#include "functions.h"
#include <stdlib.h>
#include <stdio.h>
#include <tchar.h>


void Conv3d_16b_ser(SHORT *psh_3D_sourse, const int i_WH, const int i_NinStack, 
					SHORT *psh_3D_dest, int *piKernel27, const int iNormShift)
{
	int i_SizeOfSlice = i_WH*i_WH;

	// Zeroing 1st and last resulting slices
	memset(psh_3D_dest, 0, i_SizeOfSlice*sizeof(SHORT)); // Zeroing 1-st image
	memset(psh_3D_dest+i_SizeOfSlice*(i_NinStack-1), 0, i_SizeOfSlice*sizeof(SHORT)); // Zeroing last image

	int i, j, k;

	SHORT *pSlice_in = psh_3D_sourse+i_SizeOfSlice,
		*pSlice_out = psh_3D_dest+i_SizeOfSlice,
		 *pLine_in, *pLine_out,
		 *pPix_in, *pPix_out;

	int *pKer_up_prev = piKernel27,
		  *pKer_up_cent = piKernel27+3,
		  *pKer_up_next = piKernel27+6,
		  *pKer_cur_prev = piKernel27+9,
		  *pKer_cur_cent = piKernel27+12,
		  *pKer_cur_next = piKernel27+15,
		  *pKer_bot_prev = piKernel27+18,
		  *pKer_bot_cent = piKernel27+21,
		  *pKer_bot_next = piKernel27+24;

	int   i_off_up_prev  = -i_SizeOfSlice-i_WH-1,
          i_off_up_cent  = -i_SizeOfSlice-1,
		  i_off_up_next  = -i_SizeOfSlice+i_WH-1,
		  i_off_cur_prev = -i_WH-1,
		  i_off_cur_cent = -1,
		  i_off_cur_next = i_WH-1,
		  i_off_bot_prev = i_SizeOfSlice-i_WH-1,
		  i_off_bot_cent = i_SizeOfSlice-1,
		  i_off_bot_next = i_SizeOfSlice+i_WH-1;

	for(k=1; k<(i_NinStack-1); pSlice_in += i_SizeOfSlice, pSlice_out += i_SizeOfSlice, k++)
	{	// slices

		// Zeroing 1st and last resulting lines
		memset(pSlice_out, 0, i_WH*sizeof(SHORT));
		memset(pSlice_out+i_WH*(i_WH-1), 0, i_WH*sizeof(SHORT));

		pLine_in = pSlice_in+i_WH;
		pLine_out = pSlice_out+i_WH;
		for (j=1; j<(i_WH-1); pLine_in += i_WH, pLine_out += i_WH, j++)
		{	// lines in slice

			// Zeroing 1st and last resulting pixels
			pLine_out[0] = pLine_out[i_WH-1] = 0;

			pPix_in = pLine_in+1;
			pPix_out = pLine_out+1;
			for(i=1; i<(i_WH-1); pPix_in++, pPix_out++, i++)
			{	// pixels in line

				*pPix_out = (SHORT)
					((
	*(pPix_in+i_off_up_prev)*  (*pKer_up_prev)+  *(pPix_in+i_off_up_prev+1)*  (*(pKer_up_prev+1))+  *(pPix_in+i_off_up_prev+2)*  (*(pKer_up_prev+2))+
	*(pPix_in+i_off_up_cent)*  (*pKer_up_cent)+  *(pPix_in+i_off_up_cent+1)*  (*(pKer_up_cent+1))+  *(pPix_in+i_off_up_cent+2)*  (*(pKer_up_cent+2))+
	*(pPix_in+i_off_up_next)*  (*pKer_up_next)+  *(pPix_in+i_off_up_next+1)*  (*(pKer_up_next+1))+  *(pPix_in+i_off_up_next+2)*  (*(pKer_up_next+2))+
	*(pPix_in+i_off_cur_prev)* (*pKer_cur_prev)+ *(pPix_in+i_off_cur_prev+1)* (*(pKer_cur_prev+1))+ *(pPix_in+i_off_cur_prev+2)* (*(pKer_cur_prev+2))+
	*(pPix_in+i_off_cur_cent)* (*pKer_cur_cent)+ *(pPix_in+i_off_cur_cent+1)* (*(pKer_cur_cent+1))+ *(pPix_in+i_off_cur_cent+2)* (*(pKer_cur_cent+2))+
	*(pPix_in+i_off_cur_next)* (*pKer_cur_next)+ *(pPix_in+i_off_cur_next+1)* (*(pKer_cur_next+1))+ *(pPix_in+i_off_cur_next+2)* (*(pKer_cur_next+2))+
	*(pPix_in+i_off_bot_prev)* (*pKer_bot_prev)+ *(pPix_in+i_off_bot_prev+1)* (*(pKer_bot_prev+1))+ *(pPix_in+i_off_bot_prev+2)* (*(pKer_bot_prev+2))+
	*(pPix_in+i_off_bot_cent)* (*pKer_bot_cent)+ *(pPix_in+i_off_bot_cent+1)* (*(pKer_bot_cent+1))+ *(pPix_in+i_off_bot_cent+2)* (*(pKer_bot_cent+2))+
	*(pPix_in+i_off_bot_next)* (*pKer_bot_next)+ *(pPix_in+i_off_bot_next+1)* (*(pKer_bot_next+1))+ *(pPix_in+i_off_bot_next+2)* (*(pKer_bot_next+2))
					) >> iNormShift);
			}
		}
	}

	return;		
}


bool Conv3d_16b_SSE(SHORT *psh_3D_sourse, const int i_WH, const int i_NinStack, 
					SHORT *psh_3D_dest, int *piKernel27, const int iNormShift,
					int *pi_WorkingLines)
{
	if(i_WH%8)
		return false;

	int i_SizeOfSlice = i_WH*i_WH;


    SHORT *pshSlice_in_prev, *pshSlice_in_curr, *pshSlice_in_next,
          *pshSlice_out;

    SHORT *pshLine_in_prev, *pshLine_in_curr, *pshLine_in_next,
          *pshLine_out, *pshEIGHT_out;

    int *piLineT1, *piLineT2, *piLineT3;
    int *piQUAD_res1, *piQUAD_res2,  *piQUAD_res3;

    __m128i m4int_res1, m4int_res2;

    int k, i, j, i_current_thread_num;

    for(k=0; k<i_NinStack; k++)
    {	// slices
        pshSlice_out = psh_3D_dest+k*i_SizeOfSlice;

        if(k==0 || k==(i_NinStack-1))
        {	// Zeroing is here for parallelizing by OpenMP
            memset(pshSlice_out, 0, i_SizeOfSlice*sizeof(SHORT));
            continue;
        }

        pshSlice_in_prev = psh_3D_sourse+(k-1)*i_SizeOfSlice;
        pshSlice_in_curr = psh_3D_sourse+k*i_SizeOfSlice;
        pshSlice_in_next = psh_3D_sourse+(k+1)*i_SizeOfSlice;

        for(i=0; i<i_WH; i++)
        {
            pshLine_out = pshSlice_out + i*i_WH;

            // We know, that 1-st and last lines are zeroed
            // "if" is inside to give the work for OpenMP threads
            if(i==0 || i==(i_WH-1)){memset(pshLine_out, 0, i_WH*sizeof(SHORT)); continue;};

            i_current_thread_num = omp_get_thread_num();

            piLineT1 = pi_WorkingLines + i_current_thread_num*3*i_WH;
            piLineT2 = pi_WorkingLines + (i_current_thread_num*3+1)*i_WH;
            piLineT3 = pi_WorkingLines + (i_current_thread_num*3+2)*i_WH;

            Conv2d_16to32_1line_SSE(pshSlice_in_prev+i*i_WH, i_WH, piKernel27,    piLineT1);
            Conv2d_16to32_1line_SSE(pshSlice_in_curr+i*i_WH, i_WH, piKernel27+9,  piLineT2);
            Conv2d_16to32_1line_SSE(pshSlice_in_next+i*i_WH, i_WH, piKernel27+18, piLineT3);

            piQUAD_res1 = piLineT1;
            piQUAD_res2 = piLineT2;
            piQUAD_res3 = piLineT3;
            pshEIGHT_out = pshLine_out;

            // Summing up 3 lines to get final result line
            for(j=0; j<i_WH; piQUAD_res1 +=8, piQUAD_res2 +=8, piQUAD_res3 +=8, pshEIGHT_out += 8, j += 8)
            {
                // Summing up 3 & shift - 1st quad
                m4int_res1 = _mm_srai_epi32(_mm_add_epi32(
                    _mm_add_epi32(_mm_castps_si128(_mm_load_ps((float *)piQUAD_res1)),
                    _mm_castps_si128(_mm_load_ps((float *)piQUAD_res2))),
                    _mm_castps_si128(_mm_load_ps((float *)piQUAD_res3))
                    ), iNormShift);

                // Summing up 3 & shift - 2nd quad
                m4int_res2 = _mm_srai_epi32(_mm_add_epi32(
                    _mm_add_epi32(_mm_castps_si128(_mm_load_ps((float *)(piQUAD_res1+4))),
                    _mm_castps_si128(_mm_load_ps((float *)(piQUAD_res2+4)))),
                    _mm_castps_si128(_mm_load_ps((float *)(piQUAD_res3+4)))
                    ), iNormShift);

                // Packing 2 32bit QUADs to 16 bit EIGHT and saving by stream
                _mm_stream_si128((__m128i *)pshEIGHT_out,
                    _mm_packs_epi32(m4int_res1,m4int_res2));
            }
        }
    }


	return true;
}

bool Conv2d_16to32_1line_SSE(SHORT *pshLine_in_Center, const int i_WH, int *piKernel9, int *piLineOut) 
{
	static const __m128i mi_zero = _mm_setzero_si128();
	static const __m128i	       //   16  15  14  13  12  10   9   8   7   6   5   4   3   2   1   0
		mi_contr_from03 = _mm_set_epi8(255,255,  7,  6,255,255,  5,  4,255,255,  3,  2,255,255,  1,  0),
		mi_contr_from47 = _mm_set_epi8(255,255, 15, 14,255,255, 13, 12,255,255, 11, 10,255,255,  9,  8);

	__m128i mi_in16, mi_in32_1, mi_in32_2,
			mi_Multd32_1, mi_Multd32_2, 
			mi_Multd32_TopLeft_prev =mi_zero,  mi_Multd32_TopRight_prev =mi_zero,    // for Top  line, zero for prolog
			mi_Multd32_CentLeft_prev=mi_zero,  mi_Multd32_CentRight_prev=mi_zero,    // for Cent line, zero for prolog
			mi_Multd32_BotLeft_prev =mi_zero,  mi_Multd32_BotRight_prev =mi_zero,    // for Bot  line, zero for prolog
			mi_SumPrev_part, mi_SumCurr_1, mi_SumCurr_2_part;

	__m128i mi_KernTopLeft  = _mm_set1_epi32(piKernel9[0]),
			mi_KernTopCent  = _mm_set1_epi32(piKernel9[1]),
			mi_KernTopRight = _mm_set1_epi32(piKernel9[2]),
			mi_KernCentLeft = _mm_set1_epi32(piKernel9[3]),
			mi_KernCentCent = _mm_set1_epi32(piKernel9[4]),
			mi_KernCentRight= _mm_set1_epi32(piKernel9[5]),
			mi_KernBotLeft  = _mm_set1_epi32(piKernel9[6]),
			mi_KernBotCent  = _mm_set1_epi32(piKernel9[7]),
			mi_KernBotRight = _mm_set1_epi32(piKernel9[8]);

	SHORT *pshLine_in_Top = pshLine_in_Center - i_WH,
		 *pshLine_in_Bot = pshLine_in_Center + i_WH;

	int i;

// Prolog: 1st EIGHT, not in loop to prevent "if"s in a loop

// TOP line
///////////
	mi_in16 = _mm_castps_si128(_mm_load_ps((float *)pshLine_in_Top));
	
	// unpacked 1st QUAD multiplied with 1st (left) coefficient
	mi_Multd32_1 = _mm_mullo_epi32(mi_in32_1=_mm_shuffle_epi8(mi_in16,mi_contr_from03), mi_KernTopLeft); 

	// unpacked 2nd QUAD multiplied with 1st (left) coefficient
	mi_Multd32_2 = _mm_mullo_epi32(mi_in32_2=_mm_shuffle_epi8(mi_in16,mi_contr_from47), mi_KernTopLeft); 

	// Sum Left + Center - 1st QUAD
	mi_SumCurr_1 = _mm_add_epi32(_mm_alignr_epi8(mi_Multd32_1,mi_Multd32_TopLeft_prev,12),_mm_mullo_epi32(mi_in32_1, mi_KernTopCent));
	mi_Multd32_TopLeft_prev = mi_Multd32_2; // saving for next loop

	// Sum Left + Center - 2nd QUAD (in PALINGR - 1st QUAD)
	mi_SumCurr_2_part = _mm_add_epi32(_mm_alignr_epi8(mi_Multd32_2,mi_Multd32_1,12),_mm_mullo_epi32(mi_in32_2, mi_KernTopCent));

	// unpacked 1st QUAD multiplied with 3rd (right) coefficient
	mi_Multd32_1 = _mm_mullo_epi32(mi_in32_1, mi_KernTopRight); 

	// unpacked 2nd QUAD multiplied with 3rd (right) coefficient, saving for next loop
	mi_Multd32_2 = _mm_mullo_epi32(mi_in32_2, mi_KernTopRight); 

	// Inplace Sum with Right - 1st QUAD
	mi_SumCurr_1 = _mm_add_epi32(_mm_alignr_epi8(mi_Multd32_2,mi_Multd32_1,4),mi_SumCurr_1);

	// In the main loop below mi_SumCurr_2 will be completed , saving - after it
	mi_Multd32_TopRight_prev = mi_Multd32_2; // saving for next loop

	// Inplace Sum with Right is NOT here - 2nd QUAD Sum will be completed in next loop

// CENTER line
//////////////
	mi_in16 = _mm_castps_si128(_mm_load_ps((float *)pshLine_in_Center));

	// unpacked 1st QUAD multiplied with 1st (left) coefficient
	mi_Multd32_1 = _mm_mullo_epi32(mi_in32_1=_mm_shuffle_epi8(mi_in16,mi_contr_from03), mi_KernCentLeft); 

	// unpacked 2nd QUAD multiplied with 1st (left) coefficient
	mi_Multd32_2 = _mm_mullo_epi32(mi_in32_2=_mm_shuffle_epi8(mi_in16,mi_contr_from47), mi_KernCentLeft); 

	// Inplace Sum Left + Center - 1st QUAD
	mi_SumCurr_1 = _mm_add_epi32(mi_SumCurr_1,
		_mm_add_epi32(_mm_alignr_epi8(mi_Multd32_1,mi_Multd32_CentLeft_prev,12),_mm_mullo_epi32(mi_in32_1, mi_KernCentCent)));
	mi_Multd32_CentLeft_prev = mi_Multd32_2; // saving for next loop

	// In place Sum Left + Center - 2nd QUAD (in PALINGR - 1st QUAD)
	mi_SumCurr_2_part = _mm_add_epi32(mi_SumCurr_2_part,
		_mm_add_epi32(_mm_alignr_epi8(mi_Multd32_2,mi_Multd32_1,12),_mm_mullo_epi32(mi_in32_2, mi_KernCentCent)));

	// unpacked 1st QUAD multiplied with 3rd (right) coefficient
	mi_Multd32_1 = _mm_mullo_epi32(mi_in32_1, mi_KernCentRight); 

	// unpacked 2nd QUAD multiplied with 3rd (right) coefficient
	mi_Multd32_2 = _mm_mullo_epi32(mi_in32_2, mi_KernCentRight); 

	// Inplase Sum with Right - 1st QUAD
	mi_SumCurr_1 = _mm_add_epi32(_mm_alignr_epi8(mi_Multd32_2,mi_Multd32_1,4),mi_SumCurr_1);

	// In the main loop below will be completing mi_SumCurr_2, saving - after it
	mi_Multd32_CentRight_prev = mi_Multd32_2; // saving for next loop

	// Inplase Sum with Right is NOT here - 2nd QUAD Sum will be completed in next loop

	// BOTTOM line
	//////////////
	mi_in16 = _mm_castps_si128(_mm_load_ps((float *)pshLine_in_Bot));

	// unpacked 1st QUAD multiplied with 1st (left) coefficient
	mi_Multd32_1 = _mm_mullo_epi32(mi_in32_1=_mm_shuffle_epi8(mi_in16,mi_contr_from03), mi_KernBotLeft); 

	// unpacked 2nd QUAD multiplied with 1st (left) coefficient
	mi_Multd32_2 = _mm_mullo_epi32(mi_in32_2=_mm_shuffle_epi8(mi_in16,mi_contr_from47), mi_KernBotLeft); 

	// Inplace Sum Left + Center - 1st QUAD
	mi_SumCurr_1 = _mm_add_epi32(mi_SumCurr_1,
		_mm_add_epi32(_mm_alignr_epi8(mi_Multd32_1,mi_Multd32_BotLeft_prev,12),_mm_mullo_epi32(mi_in32_1, mi_KernBotCent)));
	mi_Multd32_BotLeft_prev = mi_Multd32_2; // saving for next loop

	// Sum Left + Center - 2nd QUAD (in PALINGR - 1st QUAD), save for next loop
	mi_SumPrev_part = _mm_add_epi32(mi_SumCurr_2_part,
		_mm_add_epi32(_mm_alignr_epi8(mi_Multd32_2,mi_Multd32_1,12),_mm_mullo_epi32(mi_in32_2, mi_KernBotCent)));

	// unpacked 1st QUAD multiplied with 3rd (right) coefficient
	mi_Multd32_1 = _mm_mullo_epi32(mi_in32_1, mi_KernBotRight); 

	// unpacked 2nd QUAD multiplied with 3rd (right) coefficient
	mi_Multd32_2 = _mm_mullo_epi32(mi_in32_2, mi_KernBotRight); 

	// Inplase Sum with Right - 1st QUAD - the last for all 3 lines (Tob, Cent & Bot), ready to store
	mi_SumCurr_1 = _mm_add_epi32(_mm_alignr_epi8(mi_Multd32_2,mi_Multd32_1,4),mi_SumCurr_1);
	_mm_store_si128((__m128i *)piLineOut, mi_SumCurr_1);

	piLineOut[0] = 0;  // zeroing 1st elemnt in line

	// In the main loop below mi_SumCurr_2 will be completed and stored, saving - after it
	mi_Multd32_BotRight_prev = mi_Multd32_2; // saving for next loop

	// Inplase Sum with Right is NOT here - 2nd QUAD Sum will be completed in next loop


	// And now - main loop, similar for prolog, but with full processing
	// It starts from 8, as 1st EIGHT was already treated in prolog
	for(i=8; i<i_WH; i += 8)
	{
		// TOP line
		///////////
		mi_in16 = _mm_castps_si128(_mm_load_ps((float *)(pshLine_in_Top+i)));

		// unpacked 1st QUAD multiplied with 1st (left) coefficient
		mi_Multd32_1 = _mm_mullo_epi32(mi_in32_1=_mm_shuffle_epi8(mi_in16,mi_contr_from03), mi_KernTopLeft); 

		// unpacked 2nd QUAD multiplied with 1st (left) coefficient
		mi_Multd32_2 = _mm_mullo_epi32(mi_in32_2=_mm_shuffle_epi8(mi_in16,mi_contr_from47), mi_KernTopLeft); 

		// Sum Left + Center - 1st QUAD using mi_Multd32_TopLef_prev saved in previous loop, re-saving it for next loop
		mi_SumCurr_1 = _mm_add_epi32(_mm_alignr_epi8(mi_Multd32_1,mi_Multd32_TopLeft_prev,12),_mm_mullo_epi32(mi_in32_1, mi_KernTopCent));
		mi_Multd32_TopLeft_prev = mi_Multd32_2; // saving for next loop

		// Sum Left + Center - 2nd QUAD (in PALINGR - 1st QUAD)
		mi_SumCurr_2_part = _mm_add_epi32(_mm_alignr_epi8(mi_Multd32_2,mi_Multd32_1,12),_mm_mullo_epi32(mi_in32_2, mi_KernTopCent));

		// unpacked 1st QUAD multiplied with 3rd (right) coefficient
		mi_Multd32_1 = _mm_mullo_epi32(mi_in32_1, mi_KernTopRight); 

		// unpacked 2nd QUAD multiplied with 3rd (right) coefficient, saving for next loop
		mi_Multd32_2 = _mm_mullo_epi32(mi_in32_2, mi_KernTopRight); 

		// Inplace Sum with Right - 1st QUAD
		mi_SumCurr_1 = _mm_add_epi32(_mm_alignr_epi8(mi_Multd32_2,mi_Multd32_1,4),mi_SumCurr_1);

		// Inplace sum with mi_SumPrev_part using mi_Multd32_TopRight_prev saved in previous loop 
		mi_SumPrev_part = _mm_add_epi32(_mm_alignr_epi8(mi_Multd32_1,mi_Multd32_TopRight_prev,4),mi_SumPrev_part);
		// Re-saving mi_Multd32_TopRight_prev for next loop
		mi_Multd32_TopRight_prev = mi_Multd32_2; // saving for next loop

		// CENTER line
		//////////////
		mi_in16 = _mm_castps_si128(_mm_load_ps((float *)(pshLine_in_Center+i)));

		// unpacked 1st QUAD multiplied with 1st (left) coefficient
		mi_Multd32_1 = _mm_mullo_epi32(mi_in32_1=_mm_shuffle_epi8(mi_in16,mi_contr_from03), mi_KernCentLeft); 

		// unpacked 2nd QUAD multiplied with 1st (left) coefficient
		mi_Multd32_2 = _mm_mullo_epi32(mi_in32_2=_mm_shuffle_epi8(mi_in16,mi_contr_from47), mi_KernCentLeft); 

		// Inplace Sum Left + Center - 1st QUAD
		mi_SumCurr_1 = _mm_add_epi32(mi_SumCurr_1,
			_mm_add_epi32(_mm_alignr_epi8(mi_Multd32_1,mi_Multd32_CentLeft_prev,12),_mm_mullo_epi32(mi_in32_1, mi_KernCentCent)));
		mi_Multd32_CentLeft_prev = mi_Multd32_2; // saving for next loop

		// Sum Left + Center - 2nd QUAD (in PALINGR - 1st QUAD)
		mi_SumCurr_2_part = _mm_add_epi32(mi_SumCurr_2_part,
			_mm_add_epi32(_mm_alignr_epi8(mi_Multd32_2,mi_Multd32_1,12),_mm_mullo_epi32(mi_in32_2, mi_KernCentCent)));

		// unpacked 1st QUAD multiplied with 3rd (right) coefficient
		mi_Multd32_1 = _mm_mullo_epi32(mi_in32_1, mi_KernCentRight); 

		// unpacked 2nd QUAD multiplied with 3rd (right) coefficient
		mi_Multd32_2 = _mm_mullo_epi32(mi_in32_2, mi_KernCentRight); 

		// Inplase Sum with Right - 1st QUAD
		mi_SumCurr_1 = _mm_add_epi32(_mm_alignr_epi8(mi_Multd32_2,mi_Multd32_1,4),mi_SumCurr_1);

		// Inplace sum with mi_SumPrev_part using mi_Multd32_TopRight_prev saved in previous loop 
		mi_SumPrev_part = _mm_add_epi32(_mm_alignr_epi8(mi_Multd32_1,mi_Multd32_CentRight_prev,4),mi_SumPrev_part);
		// Re-saving mi_Multd32_TopRight_prev for next loop
		mi_Multd32_CentRight_prev = mi_Multd32_2; 

		// BOTTOM line
		//////////////
		mi_in16 = _mm_castps_si128(_mm_load_ps((float *)(pshLine_in_Bot+i)));

		// unpacked 1st QUAD multiplied with 1st (left) coefficient
		mi_Multd32_1 = _mm_mullo_epi32(mi_in32_1=_mm_shuffle_epi8(mi_in16,mi_contr_from03), mi_KernBotLeft); 

		// unpacked 2nd QUAD multiplied with 1st (left) coefficient
		mi_Multd32_2 = _mm_mullo_epi32(mi_in32_2=_mm_shuffle_epi8(mi_in16,mi_contr_from47), mi_KernBotLeft); 

		// Inplace Sum Left + Center - 1st QUAD
		mi_SumCurr_1 = _mm_add_epi32(mi_SumCurr_1,
			_mm_add_epi32(_mm_alignr_epi8(mi_Multd32_1,mi_Multd32_BotLeft_prev,12),_mm_mullo_epi32(mi_in32_1, mi_KernBotCent)));
		mi_Multd32_BotLeft_prev = mi_Multd32_2; // saving for next loop

		// Sum Left + Center - 2nd QUAD (in PALINGR - 1st QUAD)
		mi_SumCurr_2_part = _mm_add_epi32(mi_SumCurr_2_part,
			_mm_add_epi32(_mm_alignr_epi8(mi_Multd32_2,mi_Multd32_1,12),_mm_mullo_epi32(mi_in32_2, mi_KernBotCent)));

		// unpacked 1st QUAD multiplied with 3rd (right) coefficient
		mi_Multd32_1 = _mm_mullo_epi32(mi_in32_1, mi_KernBotRight); 

		// unpacked 2nd QUAD multiplied with 3rd (right) coefficient
		mi_Multd32_2 = _mm_mullo_epi32(mi_in32_2, mi_KernBotRight); 

		// Inplase Sum with Right - 1st QUAD - the last for all 3 lines (Tob, Cent & Bot), ready to store
		mi_SumCurr_1 = _mm_add_epi32(_mm_alignr_epi8(mi_Multd32_2,mi_Multd32_1,4),mi_SumCurr_1);
		_mm_store_si128((__m128i *)(piLineOut+i), mi_SumCurr_1);

		// Inplace sum with mi_SumPrev_part using mi_Multd32_TopRight_prev saved in previous loop 
		mi_SumPrev_part = _mm_add_epi32(_mm_alignr_epi8(mi_Multd32_1,mi_Multd32_BotRight_prev,4),mi_SumPrev_part);
		// Re-saving mi_Multd32_TopRight_prev for next loop
		mi_Multd32_BotRight_prev = mi_Multd32_2; // saving for next loop

		// Here we can finally save previous QUAD - it is ready !
		_mm_store_si128((__m128i *)(piLineOut+i-4), mi_SumPrev_part);

		mi_SumPrev_part = mi_SumCurr_2_part;
	}

// Epilog: treating last QUAD

	mi_SumPrev_part = _mm_add_epi32(_mm_alignr_epi8(mi_zero,mi_Multd32_TopRight_prev,4),mi_SumPrev_part);
	mi_SumPrev_part = _mm_add_epi32(_mm_alignr_epi8(mi_zero,mi_Multd32_CentRight_prev,4),mi_SumPrev_part);
	mi_SumPrev_part = _mm_add_epi32(_mm_alignr_epi8(mi_zero,mi_Multd32_BotRight_prev,4),mi_SumPrev_part);

	// Here i is advanced to boundary of the line
	_mm_store_si128((__m128i *)(piLineOut+i-4), mi_SumPrev_part);
	*(piLineOut + i_WH -1) = 0; // zeroing the last pixel

	return true;
}

int Diff3d_16b(SHORT *psh_3D_1, const int i_WH, const int i_NinStack, SHORT *psh_3D_2)
{
	int i, i_Diffs=0, i_DiffMax=-1, i_CurrDiff;
	int i_diff_1st=-1;

	for(i=0; i<i_WH*i_WH*i_NinStack; i++)
	{
		if(0 != (i_CurrDiff = abs(psh_3D_1[i] - psh_3D_2[i])))
		{
			if(i_diff_1st == -1)
				i_diff_1st=i;
			
			if(i_CurrDiff > i_DiffMax)
				i_DiffMax = i_CurrDiff;

			i_Diffs++;
		}
	}
	
	if(i_Diffs)
	{
		int i_SlNum = i_diff_1st/(i_WH*i_WH),
			i_OffOnSlice = i_diff_1st - i_SlNum*i_WH*i_WH,
			i_LineNum = i_OffOnSlice/i_WH,
			i_PixNum = i_OffOnSlice - i_WH*i_LineNum;

		printf("There are %d diffs; maximal diff=%d\n1st on slice %d, line %d, pixel %d: %d <=> %d\n\n",
			i_Diffs,i_DiffMax, i_SlNum,i_LineNum,i_PixNum,
			(int)psh_3D_1[i_diff_1st],(int)psh_3D_2[i_diff_1st]);
	}
	else
		printf("There are no diffs\n\n");

	return i_Diffs;
}
