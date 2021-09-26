// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// sshd_test_pw.c
// Wrapper to inject test password data for sshd PAM authentication
//
// This wrapper implements custom versions of getpwnam, getpwnam_r,
// getspnam and getspnam_r. These functions first call their real
// libc versions, then check if the requested user matches test user
// specified in env variable TEST_USER and if so replace the password
// with crypted() value of TEST_PASSWD env variable.
//
// Compile:
// gcc -Wall -shared -o sshd_test_pw.so -fPIC sshd_test_pw.c
//
// Compile with debug:
// gcc -DVERBOSE -Wall -shared -o sshd_test_pw.so -fPIC sshd_test_pw.c
//
// Run sshd:
// LD_PRELOAD="sshd_test_pw.so" TEST_USER="..." TEST_PASSWD="..." sshd ...

// +build ignore

#define _GNU_SOURCE
#include <string.h>
#include <pwd.h>
#include <shadow.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>

#ifdef VERBOSE
#define DEBUG(X...) fprintf(stderr, X)
#else
#define DEBUG(X...) while (0) { }
#endif

/* crypt() password */
static char *
pwhash(char *passwd) {
  return strdup(crypt(passwd, "$6$"));
}

/* Pointers to real functions in libc */
static struct passwd * (*real_getpwnam)(const char *) = NULL;
static int (*real_getpwnam_r)(const char *, struct passwd *, char *, size_t, struct passwd **) = NULL;
static struct spwd * (*real_getspnam)(const char *) = NULL;
static int (*real_getspnam_r)(const char *, struct spwd *, char *, size_t, struct spwd **) = NULL;

/* Cached test user and test password */
static char *test_user = NULL;
static char *test_passwd_hash = NULL;

static void
init(void) {
  /* Fetch real libc function pointers */
  real_getpwnam = dlsym(RTLD_NEXT, "getpwnam");
  real_getpwnam_r = dlsym(RTLD_NEXT, "getpwnam_r");
  real_getspnam = dlsym(RTLD_NEXT, "getspnam");
  real_getspnam_r = dlsym(RTLD_NEXT, "getspnam_r");
  
  /* abort if env variables are not defined */
  if (getenv("TEST_USER") == NULL || getenv("TEST_PASSWD") == NULL) {
    fprintf(stderr, "env variables TEST_USER and TEST_PASSWD are missing\n");
    abort();
  }

  /* Fetch test user and test password from env */
  test_user = strdup(getenv("TEST_USER"));
  test_passwd_hash = pwhash(getenv("TEST_PASSWD"));

  DEBUG("sshd_test_pw init():\n");
  DEBUG("\treal_getpwnam: %p\n", real_getpwnam);
  DEBUG("\treal_getpwnam_r: %p\n", real_getpwnam_r);
  DEBUG("\treal_getspnam: %p\n", real_getspnam);
  DEBUG("\treal_getspnam_r: %p\n", real_getspnam_r);
  DEBUG("\tTEST_USER: '%s'\n", test_user);
  DEBUG("\tTEST_PASSWD: '%s'\n", getenv("TEST_PASSWD"));
  DEBUG("\tTEST_PASSWD_HASH: '%s'\n", test_passwd_hash);
}

static int
is_test_user(const char *name) {
  if (test_user != NULL && strcmp(test_user, name) == 0)
    return 1;
  return 0;
}

/* getpwnam */

struct passwd *
getpwnam(const char *name) {
  struct passwd *pw;

  DEBUG("sshd_test_pw getpwnam(%s)\n", name);
  
  if (real_getpwnam == NULL)
    init();
  if ((pw = real_getpwnam(name)) == NULL)
    return NULL;

  if (is_test_user(name))
    pw->pw_passwd = strdup(test_passwd_hash);
      
  return pw;
}

/* getpwnam_r */

int
getpwnam_r(const char *name,
	   struct passwd *pwd,
	   char *buf,
	   size_t buflen,
	   struct passwd **result) {
  int r;

  DEBUG("sshd_test_pw getpwnam_r(%s)\n", name);
  
  if (real_getpwnam_r == NULL)
    init();
  if ((r = real_getpwnam_r(name, pwd, buf, buflen, result)) != 0 || *result == NULL)
    return r;

  if (is_test_user(name))
    pwd->pw_passwd = strdup(test_passwd_hash);
  
  return 0;
}

/* getspnam */

struct spwd *
getspnam(const char *name) {
  struct spwd *sp;

  DEBUG("sshd_test_pw getspnam(%s)\n", name);
  
  if (real_getspnam == NULL)
    init();
  if ((sp = real_getspnam(name)) == NULL)
    return NULL;

  if (is_test_user(name))
    sp->sp_pwdp = strdup(test_passwd_hash);
  
  return sp;
}

/* getspnam_r */

int
getspnam_r(const char *name,
	   struct spwd *spbuf,
	   char *buf,
	   size_t buflen,
	   struct spwd **spbufp) {
  int r;

  DEBUG("sshd_test_pw getspnam_r(%s)\n", name);
  
  if (real_getspnam_r == NULL)
    init();
  if ((r = real_getspnam_r(name, spbuf, buf, buflen, spbufp)) != 0)
    return r;

  if (is_test_user(name))
    spbuf->sp_pwdp = strdup(test_passwd_hash);
  
  return r;
}
