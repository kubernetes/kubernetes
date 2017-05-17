//Package cmd uses raw cmd to get the per-pid gpu sm util and mem util
package cmd

import (
	"context"
	"os"
	"os/exec"
	"strings"
	"time"
	"sync"
	"github.com/golang/glog"
)

type CMDGPUMonitor struct {
	utillock sync.RWMutex
	fbSizelock sync.RWMutex
	// GPUUtils first key is pid, second key is device id, value is two-elemet util slice, first element is sm util
	// second element is mem util
	GPUUtils map[string]map[string][]string
	// GPUFBSize first key is pid, second key is device id, value is fb size(unit is MB)
	GPUFBSize map[string]map[string]string
}

func NewGPUMonitor() *CMDGPUMonitor{
	return &CMDGPUMonitor{
		GPUUtils: make(map[string]map[string][]string),
		GPUFBSize: make(map[string]map[string]string),
	}
}

// IsCmdExist checks whether nvidia-smi cmd exist or not
func (self *CMDGPUMonitor) isCmdExist() bool {
	_, err := exec.LookPath("nvidia-smi")
	if err != nil {
		return false
	}

	return true
}

// SetGPUUtils runs nvidia-smi command and set per pid sm util and mem util
func (self *CMDGPUMonitor) setGPUUtils() (error) {
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)

	defer cancel()

	cmd := exec.CommandContext(ctx, "nvidia-smi", "pmon", "-c", "1")

	env := os.Environ()
	env = append(env, "NVSMI_SHOW_ALL_DEVICES=1")
	cmd.Env = env

	out, err := cmd.Output()

	if err != nil {
		return err
	}

	// sample output
	/*# gpu     pid  type    sm   mem   enc   dec   command
	# Idx       #   C/G     %     %     %     %   name
		0       -     -     -     -     -     -   -
		1       -     -     -     -     -     -   -
		2       -     -     -     -     -     -   -
		3       -     -     -     -     -     -   -
		4       -     -     -     -     -     -   -
		5       -     -     -     -     -     -   -
		6       -     -     -     -     -     -   -
		7       -     -     -     -     -     -   -
		8       -     -     -     -     -     -   -
		9       -     -     -     -     -     -   -
		10   64756     C     0     0     0     0   pulpf
		11       -     -     -     -     -     -   -
		12       -     -     -     -     -     -   -
		13       -     -     -     -     -     -   -
		14       -     -     -     -     -     -   -
		15 1426541     C    66    26     0     0   python */
	
	newPids := make(map[string]struct{})
    self.utillock.Lock()

	for c, line := range strings.Split(string(out), "\n") {
		vals := strings.Fields(line)
		if c < 2 || len(vals) != 8 {
			continue
		}

		if vals[1] == "-" {
			continue
		}
		newPids[vals[1]] = struct{}{}

		utilmap, ok := self.GPUUtils[vals[1]]

		if !ok {
			// new process emerge
			self.GPUUtils[vals[1]] = make(map[string][]string)
			utilmap, _ = self.GPUUtils[vals[1]]
		}

		if len(utilmap[vals[0]]) == 0 {
			// new device util metric
			utilmap[vals[0]] = make([]string,2,2)
			
		}
		utilmap[vals[0]][0]=vals[3]
	    utilmap[vals[0]][1]=vals[4]
		
	}

    //delete none-exist entry
	for pid := range self.GPUUtils{
		if _, ok := newPids[pid]; !ok{
			delete(self.GPUUtils,pid)
		}
	}


	self.utillock.Unlock()

	return nil
}

// GetGPUFBSize runs nvidia-smi command and get fb utilization
func (self *CMDGPUMonitor) setGPUFBSize() error {
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)

	defer cancel()

	cmd := exec.CommandContext(ctx, "nvidia-smi", "pmon", "-c", "1", "-s", "m")

	env := os.Environ()
	env = append(env, "NVSMI_SHOW_ALL_DEVICES=1")
	cmd.Env = env

	out, err := cmd.Output()

	if err != nil {
		return err
	}

	// sample output
	/*# gpu     pid  type    fb   command
	# Idx       #   C/G    MB   name
	    0       -     -     -   -
	    1       -     -     -   -
	    2       -     -     -   -
	    3       -     -     -   -
	    4       -     -     -   -
	    5       -     -     -   -
	    6       -     -     -   -
	    7       -     -     -   -
	    8       -     -     -   -
	    9       -     -     -   -
	   10   38148     C   284   pulpf
	   11       -     -     -   -
	   12       -     -     -   -
	   13       -     -     -   -
	   14       -     -     -   -
	   15       -     -     -   - */
    newPids := make(map[string]struct{})
	self.fbSizelock.Lock()

	for c, line := range strings.Split(string(out), "\n") {
		vals := strings.Fields(line)
		if c < 2 || len(vals) != 5 {
			continue
		}

		if vals[1] == "-" {
			continue
		}

		newPids[vals[1]]= struct{}{}

		fbmap, ok := self.GPUFBSize[vals[1]]

		if !ok{
			// a new process emerge
			self.GPUFBSize[vals[1]] = make(map[string]string)
			fbmap, _ = self.GPUFBSize[vals[1]]
		}

		fbmap[vals[0]]=vals[3]
	}

	for pid := range self.GPUFBSize{
		if _, ok := newPids[pid]; !ok{
			delete(self.GPUFBSize,pid)
		}
	}


	self.fbSizelock.Unlock()

	return nil
}

func (self *CMDGPUMonitor) Start(){

	if !self.isCmdExist() {
		glog.Warning("there is no nvidia-smi command in the PATH")
		return
	}

	glog.V(3).Info("Start monitoring GPU")
    
	// get gpu info every 10 seconds
	ticker := time.Tick(10*time.Second)

	for {
		select {
			case t := <- ticker:
			start := time.Now()

			// do the collecting
			wg := sync.WaitGroup{}
            wg.Add(2)

			go func(){
				self.setGPUUtils()
				wg.Done()
			}()

			go func(){
				self.setGPUFBSize()
				wg.Done()
			}()

			wg.Wait()
			glog.Infof("gpu map %v %v",self.GPUFBSize,self.GPUUtils)
			
			duration := time.Since(start)

			if duration >= 2 * time.Second{
				glog.V(3).Infof("GPU monitoring(%d) took %s",t.Unix(),duration)
			}
		}
	}
}

func (self *CMDGPUMonitor) GetGPUFbSize(pid string) map[string]string {
	self.fbSizelock.RLock()
	defer self.fbSizelock.RUnlock()

	if res, ok := self.GPUFBSize[pid]; ok{
		return res
	}

	return nil
}


func (self *CMDGPUMonitor) GetGPUUtil(pid string) map[string][]string {
	self.utillock.RLock()
	defer self.utillock.RUnlock()

	if res, ok := self.GPUUtils[pid]; ok{
		return res
	}

	return nil
}
