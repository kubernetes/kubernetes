package discovery

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os/exec"
	"strconv"
	"strings"

	"k8s.io/kubernetes/cluster/addons/iso-client/coreaffinity/cputopology"
)

func DiscoverTopology() (*cputopology.CPUTopology, error) {
	isolatedCPUs, err := isolcpus()
	if err != nil {
		return nil, err
	}
	lscpuOutput, err := lscpuExecution()
	if err != nil {
		return nil, err
	}
	return cpuParse(lscpuOutput, isolatedCPUs)
}

func lscpuExecution() (string, error) {
	var stdout bytes.Buffer

	cmd := exec.Command("lscpu", "-p")

	cmd.Stdout = &stdout
	err := cmd.Run()
	if err != nil {
		return "", fmt.Errorf("Cannot run lscpu: %q", err)
	}
	return stdout.String(), nil
}

func isolcpus() ([]int, error) {
	data, err := ioutil.ReadFile("/proc/cmdline")
	if err != nil {
		return nil, fmt.Errorf("Cannot read /proc/cmdline: %q", err)
	}
	return retriveIsolcpus(string(data))
}

func retriveIsolcpus(data string) ([]int, error) {
	var isolcpusValues []string
	var cpus []int

	data = strings.Trim(data, "\n")

	for _, parameter := range strings.Split(data, " ") {
		parameterParts := strings.Split(parameter, "=")
		if len(parameterParts) != 2 {
			continue
		}
		key := parameterParts[0]
		if key != "isolcpus" {
			continue
		}
		isolcpusValues = append(isolcpusValues, strings.Split(parameterParts[1], ",")...)
	}

	for _, cpu := range isolcpusValues {
		parsedCPU, err := strconv.Atoi(cpu)
		if err != nil {
			return nil, fmt.Errorf("Cannot retrive isolcpu from list: %q", err)
		}
		cpus = append(cpus, parsedCPU)
	}
	return cpus, nil
}

func cpuParse(lscpuStringOutput string, isolatedCPUs []int) (*cputopology.CPUTopology, error) {
	lscpuRawSlice := strings.Split(lscpuStringOutput, "\n")
	var cpuTopology cputopology.CPUTopology

	for _, line := range lscpuRawSlice {
		if strings.Index(line, "#") != -1 || strings.Compare(line, "") == 0 {
			continue
		}

		var cpuid, core, socket int
		_, err := fmt.Sscanf(line, "%d,%d,%d", &cpuid, &core, &socket)
		if err != nil {
			return nil, fmt.Errorf("Cannot parse lscpu output: %q", err)
		}

		cpuTopology.CPU = append(cpuTopology.CPU, cputopology.CPU{
			SocketID:   socket,
			CoreID:     core,
			CPUID:      cpuid,
			IsIsolated: isCPUIsolated(cpuid, isolatedCPUs),
		})

	}

	return &cpuTopology, nil
}

func isCPUIsolated(cpuid int, isolatedCPUs []int) bool {
	for _, isolatedCPU := range isolatedCPUs {
		if cpuid == isolatedCPU {
			return true
		}
	}
	return false
}
