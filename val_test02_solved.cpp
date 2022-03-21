// g++ -g val_test02_solved.cpp -o val_test02_solved && valgrind --leak-check=full ./val_test02_solved

// ERRORS: even if valgrind does not complain, the orginal code does not initialize variables x[i] with i > 4
//         .. this is bad if we need those variables to have certain values
// CORRECTIONS: must let i run from 0 to 9 included, so variables are initialized with the values we want

# include <cstdlib>
# include <iostream>

using namespace std;

void junk_data ( );
int main ( );

//****************************************************************************80

int main ( )

//****************************************************************************80
//
//  Purpose:
//
//    MAIN is the main program for TEST01.
//
//  Discussion:
//
//    TEST02 has some uninitialized data.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    19 May 2011
//
{
  cout << "\n";
  cout << "TEST02:\n";
  cout << "  C++ version\n";
  cout << "  A sample code for analysis by VALGRIND.\n";

  junk_data ( );
//
//  Terminate.
//
  cout << "\n";
  cout << "TEST02\n";
  cout << "  Normal end of execution.\n";

  return 0;
}
//****************************************************************************80

void junk_data ( )

//****************************************************************************80
//
//  Purpose:
//
//    JUNK_DATA has some uninitialized variables.
//
//  Discussion:
//
//    VALGRIND's MEMCHECK program monitors uninitialized variables, but does
//    not complain unless such a variable is used in a way that means its
//    value affects the program's results, that is, the value is printed,
//    or computed with.  Simply copying the unitialized data to another variable
//    is of no concern.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    18 May 2011
//
{
  int i;
  int *x;

  x = new int[10];
//
//  X = { 0, 1, 2, 3, 4, ?a, ?b, ?c, ?d, ?e }.
//
  for ( i = 0; i < 10; i++ )  // must let i run from 0 to 9 included, so variables are initialized 
  {
    x[i] = i;
  }
//
//  Copy some values.
//  X = { 0, 1, ?c, 3, 4, ?b, ?b, ?c, ?d, ?e }.
//
  x[2] = x[7];
  x[5] = x[6];
//
//  Modify some uninitialized entries.
//  Memcheck doesn't seem to care about this.
//
  for ( i = 0; i < 10; i++ )
  {
    x[i] = 2 * x[i];
  }
//
//  Print X.
//
  for ( i = 0; i < 10; i++ )
  {
    cout << "  " << i << "  " << x[i] << "\n";
  }

  delete [] x;

  return;
}
