// +build !windows

package main

import (
	"encoding/json"
	"strings"

	"github.com/go-check/check"
)

func (s *DockerSuite) TestBuildResourceConstraintsAreUsed(c *check.C) {
	testRequires(c, CpuCfsQuota)
	name := "testbuildresourceconstraints"

	ctx, err := fakeContext(`
	FROM hello-world:frozen
	RUN ["/hello"]
	`, map[string]string{})
	if err != nil {
		c.Fatal(err)
	}

	dockerCmdInDir(c, ctx.Dir, "build", "--no-cache", "--rm=false", "--memory=64m", "--memory-swap=-1", "--cpuset-cpus=0", "--cpuset-mems=0", "--cpu-shares=100", "--cpu-quota=8000", "-t", name, ".")

	out, _ := dockerCmd(c, "ps", "-lq")

	cID := strings.TrimSpace(out)

	type hostConfig struct {
		Memory     int64
		MemorySwap int64
		CpusetCpus string
		CpusetMems string
		CpuShares  int64
		CpuQuota   int64
	}

	cfg, err := inspectFieldJSON(cID, "HostConfig")
	if err != nil {
		c.Fatal(err)
	}

	var c1 hostConfig
	if err := json.Unmarshal([]byte(cfg), &c1); err != nil {
		c.Fatal(err, cfg)
	}
	if c1.Memory != 67108864 || c1.MemorySwap != -1 || c1.CpusetCpus != "0" || c1.CpusetMems != "0" || c1.CpuShares != 100 || c1.CpuQuota != 8000 {
		c.Fatalf("resource constraints not set properly:\nMemory: %d, MemSwap: %d, CpusetCpus: %s, CpusetMems: %s, CpuShares: %d, CpuQuota: %d",
			c1.Memory, c1.MemorySwap, c1.CpusetCpus, c1.CpusetMems, c1.CpuShares, c1.CpuQuota)
	}

	// Make sure constraints aren't saved to image
	dockerCmd(c, "run", "--name=test", name)

	cfg, err = inspectFieldJSON("test", "HostConfig")
	if err != nil {
		c.Fatal(err)
	}
	var c2 hostConfig
	if err := json.Unmarshal([]byte(cfg), &c2); err != nil {
		c.Fatal(err, cfg)
	}
	if c2.Memory == 67108864 || c2.MemorySwap == -1 || c2.CpusetCpus == "0" || c2.CpusetMems == "0" || c2.CpuShares == 100 || c2.CpuQuota == 8000 {
		c.Fatalf("resource constraints leaked from build:\nMemory: %d, MemSwap: %d, CpusetCpus: %s, CpusetMems: %s, CpuShares: %d, CpuQuota: %d",
			c2.Memory, c2.MemorySwap, c2.CpusetCpus, c2.CpusetMems, c2.CpuShares, c2.CpuQuota)
	}

}
