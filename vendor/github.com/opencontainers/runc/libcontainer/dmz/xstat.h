#ifndef XSTAT_H
#define XSTAT_H

// Some old-kernels (like centos-7) don't have statx() defined in linux/stat.h. We can't include
// sys/stat.h because it creates conflicts, so let's just define what we need here and be done with
// this.
// TODO (rata): I'll probably submit a patch to nolibc upstream so we can remove this hack in the
// future.
#include <linux/stat.h>  /* for statx() */

#ifndef STATX_BASIC_STATS
#include "linux/stat.h"
#endif // STATX_BASIC_STATS

#endif // XSTAT_H
