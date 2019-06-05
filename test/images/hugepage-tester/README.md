# Hugepage tester

> Simple container with a util for testing huge page allocation

Usage ./hugetlb-tester <size> <page-size> <path>

- <size> is amount of bytes you want to allocate using huge pages
- <pagesize> is the given page size in bytes
- <path> is a path to a file on a hugetlbfs

If the program returns 0, all allocations were successful

Example using 64MiB of 2MiB pages, with hugetlbfs mounted in /dev/hugetlb
./hugetlb-tester $((1<<26)) $((1<<21)) /dev/hugetlb/file
