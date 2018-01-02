dnl Checks for thread-local storage support.
dnl
dnl Taken from the openvswitch config code (Apache 2.0 License)
dnl with some local modifications. Does not include <threads.h>
dnl as this does not currently exist on GCC.
dnl Checks whether the compiler and linker support the C11
dnl thread_local macro from <threads.h>, and if so defines
dnl HAVE_THREAD_LOCAL.  If not, checks whether the compiler and linker
dnl support the GCC __thread extension, and if so defines
dnl HAVE___THREAD.
AC_DEFUN([CT_CHECK_TLS],
  [AC_CACHE_CHECK(
     [whether $CC has <threads.h> that supports thread_local],
     [ct_cv_thread_local],
     [AC_LINK_IFELSE(
        [AC_LANG_PROGRAM([ static thread_local int var;], [return var;])],
        [ct_cv_thread_local=yes],
        [ct_cv_thread_local=no])])
   if test $ct_cv_thread_local = yes; then
     AC_DEFINE([HAVE_THREAD_LOCAL], [1],
               [Define to 1 if the C compiler and linker supports the C11
                thread_local macro defined in <threads.h>.])
   else
     AC_CACHE_CHECK(
       [whether $CC supports __thread],
       [ct_cv___thread],
       [AC_LINK_IFELSE(
          [AC_LANG_PROGRAM([static __thread int var;], [return var;])],
          [ct_cv___thread=yes],
          [ct_cv___thread=no])])
     if test $ct_cv___thread = yes; then
       AC_DEFINE([HAVE___THREAD], [1],
                 [Define to 1 if the C compiler and linker supports the
                  GCC __thread extensions.])
     fi
   fi])

