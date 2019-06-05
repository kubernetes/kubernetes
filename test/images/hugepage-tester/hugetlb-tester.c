#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#define PROTECTION (PROT_READ | PROT_WRITE)

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif

#ifndef MAP_HUGE_MASK
#define MAP_HUGE_MASK 0x3f
#endif

#define ADDR (void *)(0x0UL)
#ifndef MAP_HUGETLB
#define MAP_HUGETLB
#endif
#define FLAGS (MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB)

static void write_bytes(char *addr, size_t length) {
  unsigned long i;

  for (i = 0; i < length; i++)
    *(addr + i) = (char)i;
}

static int read_bytes(char *addr, size_t length) {
  unsigned long i;

  for (i = 0; i < length; i++)
    if (*(addr + i) != (char)i) {
      printf("Mismatch at %lu\n", i);
      return 1;
    }
  return 0;
}

#define suffix_len 4
const char *const suffixes[suffix_len] = {"B", "KiB", "MiB", "GiB"};
void pretty_bytes(char *buf, size_t bytes) {
  int s = 0; // which suffix to use
  size_t rel = bytes;
  while (rel >= 1024 && s < 4)
    s++, rel >>= 10;
  sprintf(buf, "%d%s", (int)rel, suffixes[s]);
}

int verify_using_mmap(size_t length, size_t hugepage_size, int flags) {
  int ret;
  void *addr;
  // using mmap
  addr = mmap(ADDR, length, PROTECTION, flags, -1, 0);
  if (addr == MAP_FAILED) {
    perror("mmap");
    exit(1);
  }

  printf("Returned address is %p\n", addr);
  write_bytes(addr, length);
  ret = read_bytes(addr, length);

  if (munmap(addr, length)) {
    perror("munmap");
    exit(1);
  }
  return ret;

  // End MMAP
}

int verify_using_hugetlb_file(size_t length, char *filename) {
  void *addr;
  int fd, ret;

  fd = open(filename, O_CREAT | O_RDWR, 0755);
  if (fd < 0) {
    perror("Open failed");
    exit(1);
  }

  addr = mmap(0x00UL, length, PROTECTION, MAP_SHARED, fd, 0);
  if (addr == MAP_FAILED) {
    perror("mmap");
    unlink(filename);
    exit(1);
  }

  printf("Returned address is %p\n", addr);
  write_bytes(addr, length);
  ret = read_bytes(addr, length);

  munmap(addr, length);
  close(fd);
  unlink(filename);
  return ret;
}

// Usage
// ./hugetlb-tester <size> <page-size> <mountpath>
// mountpath is the path of the mounted hugetlbfs
// Both pagesize and size is measured in bytes
int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(
        stderr,
        "Usage "
        "./hugetlb-tester <size> <page-size> <path>\n"
        "- <size> is amount of bytes you want to allocate using huge pages\n"
        "- <pagesize> is the given page size in bytes\n"
        "- <path> is a path to a file on a hugetlbfs\n\n"
        "If the program returns 0, all allocations were successful\n\n"
        "Example using 64MiB of 2MiB pages, with hugetlbfs mounted in "
        "/dev/hugetlb\n"
        "./hugetlb-tester $((1<<26)) $((1<<21)) /dev/hugetlb/file\n");
    return 1;
  }
  int ret;
  int flags = FLAGS;
  const size_t length = strtol(argv[1], NULL, 10);
  const size_t pagesize = strtol(argv[2], NULL, 10);
  char *mountpath = argv[3];
  int shift = 0;
  int rel_pagesize = pagesize;
  while (!(rel_pagesize & 1)) {
    shift++, rel_pagesize >>= 1;
  }
  flags |= (shift & MAP_HUGE_MASK) << MAP_HUGE_SHIFT;
  char human_length[10], human_pagesize[10];
  pretty_bytes(human_length, length);
  pretty_bytes(human_pagesize, pagesize);
  printf("Mapping %s on %ld pages with %s\n", human_length, (length / pagesize),
         human_pagesize);

  if (length % (1 << shift)) {
    printf("You are not allocation a multiple of the pagesize, strange "
           "behavior may occur\n");
  }
  ret = verify_using_mmap(length, pagesize, flags);
  ret += verify_using_hugetlb_file(length, mountpath);
  return ret;
}
