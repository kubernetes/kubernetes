/*
 * Copyright 2016 SUSE LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#if !defined(CMSG_H)
#define CMSG_H

#include <sys/types.h>

/* TODO: Implement this properly with MSG_PEEK. */
#define TAG_BUFFER 4096

/* This mirrors Go's (*os.File). */
struct file_t {
	char *name;
	int fd;
};

struct file_t recvfd(int sockfd);
ssize_t sendfd(int sockfd, struct file_t file);

#endif /* !defined(CMSG_H) */
