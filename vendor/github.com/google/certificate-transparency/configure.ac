AC_PREREQ([2.68])
AC_INIT([certificate-transparency], [0.9], [certificate-transparency@googlegroups.com])
AM_INIT_AUTOMAKE([-Wall -Werror foreign subdir-objects])
AC_CONFIG_SRCDIR([cpp/server/ct-server.cc])
AC_CONFIG_HEADERS([cpp/config.h])
AC_CONFIG_MACRO_DIR([m4])
AM_SILENT_RULES([yes])
AC_LANG([C++])

AC_ARG_ENABLE(hardening,
              AS_HELP_STRING([--disable-hardening], [Use C++ compiler flags which produce a hardened binary]))

GMOCK_DIR="${GMOCK_DIR=/usr/src/gmock}"
AC_ARG_VAR([GMOCK_DIR], [directory containing Google Mock])
GTEST_DIR="${GTEST_DIR="$GMOCK_DIR/gtest"}"
AC_ARG_VAR([GTEST_DIR], [directory containing Google Test])
INSTALL_DIR="${INSTALL_DIR=$srcdir/install}"
AC_ARG_VAR([INSTALL_DIR], [location of installed dependencies])

# Checks for programs.
AC_PROG_AWK
AC_PROG_CXX
AC_PROG_CC
AM_PROG_CC_C_O
AC_PROG_CPP
AC_PROG_LN_S
AC_PROG_RANLIB
AM_PROG_AR
AX_CXX_COMPILE_STDCXX_11
AC_CHECK_PROGS([ANT], [ant])

PKG_CHECK_MODULES([json_c], [json-c])
PKG_CHECK_MODULES([evhtp], [evhtp])
PKG_CHECK_MODULES([icu], [icu-uc])

if test "x${enable_hardening}" != "xno"; then
  common_harden_copts="-fstack-protector-all -fPIC -Wa,--noexecstack -Wformat -Wformat-security"
  clang_harden_copts="-Qunused-arguments $common_harden_copts"
  gcc_harden_copts="$common_harden_copts"
  #TODO(alcutter): perform magic to determine if c++/cc are indeed clang.
  AS_CASE([$CXX],
          [c++], [AS_VAR_APPEND([CXXFLAGS], [" $clang_harden_copts"])], # probably Clang on FBSD/OSX
          [clang++*], [AS_VAR_APPEND([CXXFLAGS], [" $clang_harden_copts"])],
          [g++*], [AS_VAR_APPEND([CXXFLAGS], [" $gcc_harden_copts"])],
          [AC_MSG_FAILURE([Hardening enabled, but we don't have hardening flags for C++ compiler $CXX])])
  AS_CASE([$CC],
          [cc], [AS_VAR_APPEND([CFLAGS], [" $clang_harden_copts"])], # probably Clang on FBSD/OSX
          [clang*], [AS_VAR_APPEND([CFLAGS], [" $clang_harden_copts"])],
          [gcc*], [AS_VAR_APPEND([CFLAGS], [" $gcc_harden_copts"])],
          [AC_MSG_FAILURE(["Hardening enabled, but we don't have hardening flags for C compiler $CC"])])
  AS_VAR_APPEND([CPPFLAGS], [" -D_FORTIFY_SOURCE=2"])
  AS_VAR_APPEND([LDFLAGS], [" -pie"])

  AC_DEFINE([ENABLE_HARDENING], [], [Hardening enabled.])
else
  AC_MSG_WARN([NOT building hardened binaries])
fi

# Checks for other tools
AC_PATH_PROG([PROTOC], [protoc],,)

# Checks for header files.
AC_HEADER_RESOLV
AC_CHECK_HEADERS([arpa/inet.h fcntl.h limits.h netinet/in.h stddef.h stdint.h stdlib.h string.h sys/socket.h sys/time.h unistd.h leveldb/filter_policy.h])
AC_CHECK_HEADER([event2/event.h],,
                [AC_MSG_ERROR([libevent headers could not be found])])
AC_CHECK_HEADER([gflags/gflags.h],,
                [AC_MSG_ERROR([gflags headers could not be found])])
AC_CHECK_HEADER([glog/logging.h],,
                [AC_MSG_ERROR([glog headers could not be found])])
AC_CHECK_HEADER([google/protobuf/message.h],,
                [AC_MSG_ERROR([protobuf headers could not be found])])
AC_CHECK_HEADER([leveldb/db.h],,
                [AC_MSG_ERROR([leveldb headers could not be found])])
AC_CHECK_HEADER([ldns/ldns.h],, [missing_ldns=yes])

# Check for working GTest/GMock.
saved_CPPFLAGS="$CPPFLAGS"
AS_UNSET([missing_gtest])
AS_UNSET([missing_gmock])
AS_VAR_APPEND([CPPFLAGS], [" -I$GTEST_DIR/include"])
AC_CHECK_HEADER([gtest/gtest.h],, [missing_gtest=1])
AS_VAR_APPEND([CPPFLAGS], [" -I$GMOCK_DIR/include"])
AC_CHECK_HEADER([gmock/gmock.h],, [missing_gmock=1])
AS_VAR_APPEND([CPPFLAGS], [" -I$GTEST_DIR"])
AC_CHECK_HEADER([$GTEST_DIR/src/gtest-all.cc],, [missing_gtest=1])
AS_VAR_APPEND([CPPFLAGS], [" -I$GMOCK_DIR"])
AC_CHECK_HEADER([$GMOCK_DIR/src/gmock-all.cc],, [missing_gmock=1])
CPPFLAGS="$saved_CPPFLAGS"
AS_IF([test -n "$missing_gtest"],
      [AC_MSG_ERROR([could not find a working Google Test])])
AS_IF([test -n "$missing_gmock"],
      [AC_MSG_ERROR([could not find a working Google Mock])])

# Checks for libraries.
AC_SEARCH_LIBS([__b64_ntop], [resolv])
AC_SEARCH_LIBS([pthread_create], [pthread])
AC_SEARCH_LIBS([dlopen], [dl])

AC_MSG_CHECKING([checking for lzma library])
AC_SEARCH_LIBS([lzma_index_size], [lzma],,, [$save_LIBS])

AC_MSG_CHECKING([checking whether we need -lrt])
AC_SEARCH_LIBS([clock_gettime], [rt],,, [$save_LIBS])

AC_MSG_CHECKING([checking for gflags library])
LIBS="-lgflags $LIBS"
AC_LINK_IFELSE([AC_LANG_PROGRAM([#include <gflags/gflags.h>], [google::ParseCommandLineFlags(NULL, NULL, true)])], [have_gflags=yes], [have_gflags=no])
AC_MSG_RESULT([$have_gflags])
AS_IF([test "x$have_gflags" = "xno"],
      [AC_MSG_ERROR([gflags library could not be found])])

AC_MSG_CHECKING([checking for glog library])
LIBS="-lglog $LIBS"
AC_LINK_IFELSE([AC_LANG_PROGRAM([#include <glog/logging.h>], [google::InitGoogleLogging(NULL)])], [have_glog=yes], [have_glog=no])
AC_MSG_RESULT([$have_glog])
AS_IF([test "x$have_glog" = "xno"],
      [AC_MSG_ERROR([glog library could not be found])])

save_LIBS="$LIBS"
AS_UNSET([LIBS])
AC_SEARCH_LIBS([snappy_compress], [snappy],,, [$save_LIBS])
AC_SEARCH_LIBS([leveldb_open], [leveldb],, [missing_leveldb=1], [$save_LIBS])
AC_SUBST([leveldb_LIBS], [$LIBS])
AS_IF([test -n "$missing_leveldb"],
      [AC_MSG_ERROR([could not find the leveldb/snappy libraries])])
LIBS="$save_LIBS"

dnl We're pretty crypto-centric, having the OpenSSL libraries in LIBS
dnl is fine.
AC_SEARCH_LIBS([CRYPTO_set_locking_callback], [crypto],, [missing_openssl=1],
               [$save_LIBS])
AC_SEARCH_LIBS([SSL_CTX_new], [ssl],, [missing_openssl=1], [$save_LIBS])
AS_IF([test -n "$missing_openssl"],
      [AC_MSG_ERROR([could not find the OpenSSL libraries])])

save_LIBS="$LIBS"
AS_UNSET([LIBS])
AC_SEARCH_LIBS([event_base_dispatch], [event],, [missing_libevent=1],
               [$save_LIBS])
AC_SEARCH_LIBS([evthread_use_pthreads], [event_pthreads],, [missing_libevent=1],
               [$save_LIBS])
AC_SEARCH_LIBS([bufferevent_openssl_socket_new], [event_openssl],,
               [missing_libevent_openssl=1], [$save_LIBS])
AS_IF([test -n "$missing_libevent"],
      [AC_MSG_ERROR([could not find the libevent libraries])])
AS_IF([test -n "$missing_libevent_openssl"],
      [AC_MSG_ERROR([could not find the libevent_openssl library])])

AC_SUBST([libevent_LIBS], [$LIBS])

LIBS="$save_LIBS"

# TCMalloc gubbins
AC_ARG_WITH([tcmalloc],
            [AS_HELP_STRING([--without-tcmalloc],
                            [disable tcmalloc for memory allocations])],
            [],
            [with_tcmalloc=yes])
AS_IF([test "x$with_tcmalloc" != xno],
      [AC_CHECK_LIB([tcmalloc], [malloc],,
                    [AC_MSG_FAILURE([no tcmalloc found (use --without-tcmalloc to disable)])])])

# Checks for typedefs, structures, and compiler characteristics.
AC_TYPE_INT32_T
AC_TYPE_INT64_T
AC_TYPE_MODE_T
AC_TYPE_SIZE_T
AC_TYPE_SSIZE_T
AC_TYPE_UINT16_T
AC_TYPE_UINT32_T
AC_TYPE_UINT64_T
AC_TYPE_UINT8_T

CT_CHECK_TLS
AC_CHECK_DECLS([INADDR_LOOPBACK], [], [], [#include <netinet/in.h>])

AC_MSG_CHECKING([whether pthread_t is a pointer])
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([
#include <pthread.h>
void TakesPointer(void*);
], [TakesPointer(pthread_self())])], [PTHREAD_T_IS_POINTER=yes],
   [PTHREAD_T_IS_POINTER=no])
AC_MSG_RESULT([$PTHREAD_T_IS_POINTER])
AS_IF([test "x$PTHREAD_T_IS_POINTER" = xyes],
      [AC_DEFINE([PTHREAD_T_IS_POINTER], [],
                 [Whether pthread_t is a pointer.])])

# Checks for library functions.
AC_FUNC_FORK
AC_CHECK_FUNCS([alarm gettimeofday memset mkdir select socket strdup strerror strtol])

# TODO(pphaneuf): We should validate that we have all the tools and
# libraries that we require here, instead of letting the compilation
# fail later during the build (sometimes in ways that are unclear to
# the user.


AM_CONDITIONAL([HAVE_ANT], [test -n "$ANT"])
AM_CONDITIONAL([HAVE_LDNS], [test -z "$missing_ldns"])
AC_DEFINE_UNQUOTED([TEST_SRCDIR], ["$srcdir"], [Top of the source directory, for tests.])
AC_SUBST([INSTALL_DIR])
AC_CONFIG_FILES([Makefile])
AC_OUTPUT
