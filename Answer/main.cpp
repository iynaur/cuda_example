#include "answer.cuh"

#include<stdio.h>
#include<stdlib.h>
#include<math.h>

int main()
{
    float *a, *b, *c;
    int N = 1<<20;
    int size = N * sizeof( float );
    /* allocate space for device copies of a, b, c */
    /* allocate space for host copies of a, b, c and setup input values */

//Allocate memory for each vector on host
    a = (float *)malloc( size );
    b = (float *)malloc( size );
    c = (float *)malloc( size );

    for( int i = 0; i < N; i++ )
    {
        a[i] = b[i] = i;
        c[i] = 0;
    }

    add(a,b,c,N);


//	printf( "c[0] = %f\n",c[0] );
    for( int i = 0; i < 10; i++ ) printf( "c[%d] = %f\n",i, c[i] );

    /* clean up */

    free(a);
    free(b);
    free(c);


    return 0;
} /* end main */
