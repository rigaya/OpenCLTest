/*****************************************************************************
 * Copyright (c) 2013-2016 Intel Corporation
 * All rights reserved.
 *
 * WARRANTY DISCLAIMER
 *
 * THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE
 * MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Intel Corporation is the author of the Materials, and requests that all
 * problem reports or change requests be submitted to it directly
 *****************************************************************************/

__kernel void convolve(__global float* pIn, __global float* pOut, int width, int pitch, int height) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x < width && y < height) {
        const float weight[] = {
            1.0f, 2.0f, 1.0f,
            2.0f, 4.0f, 2.0f,
            1.0f, 2.0f, 1.0f,
        };
        float sump = 0.0f;
        float sumw = 0.0f;
        int iw = 0;
        for (int j = -1; j <= 1; j++) {
            int yj = clamp(y+j, 0, height-1);
            for (int i = -1; i <= 1; i++) {
                int xi = clamp(x+i, 0, width-1);
                float w = weight[iw];
                sumw += w;
                sump += w * pIn[yj * pitch + xi];
                iw++;
            }
        }
        pOut[y * pitch + x] = sump / sumw;
    }
}
