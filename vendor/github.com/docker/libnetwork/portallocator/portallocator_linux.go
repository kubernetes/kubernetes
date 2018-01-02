package portallocator

import (
	"bufio"
	"fmt"
	"os"
)

func getDynamicPortRange() (start int, end int, err error) {
	const portRangeKernelParam = "/proc/sys/net/ipv4/ip_local_port_range"
	portRangeFallback := fmt.Sprintf("using fallback port range %d-%d", DefaultPortRangeStart, DefaultPortRangeEnd)
	file, err := os.Open(portRangeKernelParam)
	if err != nil {
		return 0, 0, fmt.Errorf("port allocator - %s due to error: %v", portRangeFallback, err)
	}

	defer file.Close()

	n, err := fmt.Fscanf(bufio.NewReader(file), "%d\t%d", &start, &end)
	if n != 2 || err != nil {
		if err == nil {
			err = fmt.Errorf("unexpected count of parsed numbers (%d)", n)
		}
		return 0, 0, fmt.Errorf("port allocator - failed to parse system ephemeral port range from %s - %s: %v", portRangeKernelParam, portRangeFallback, err)
	}
	return start, end, nil
}
