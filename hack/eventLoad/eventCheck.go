package main

import (
	"github.com/spf13/cobra"
	"fmt"
	"os"
	"bufio"
	"strings"
	"log"
	"strconv"
	"io/ioutil"
	"path/filepath"
	//"time"
	"sort"
)

const (
	message = `List 0: "etcd3/watcher/transform/curObj", "etcd3/watcher/transform/oldObj"
List 1: "etcd3/watcher/processEvent"
List 2: "watch_cache/processEvent
List 3: "cacher/dispatchEvent"
List 4: "cacher/add0"
List 5: "cacher/add1", "cacher/add2", "cacher/add/case3"
List 6: "cacher/send0"
List 7: "cacher/send1"
List 8: "cacher/send2"
List 9: scheduler reflecter
List 10: kcm reflecter
List 11: "watch/ServeHTTP""
Please find details here: https://github.com/kubernetes/kubernetes/pull/61067
`
)


var (
	eventCheckCmd = &cobra.Command{
		Short: "A tool to check if there is any missing event based on resourceVersion.",
		Long: message,
		Run: func(cmd *cobra.Command, args []string) {
			runEventCheck()
		},
	}
	ecOpts = eventCheckOpts{}
	rvLists [][]int
	eventLists [][]eventTrackerEntry
	apiEventList []eventTrackerEntry
	schedulerEventList []eventTrackerEntry
	kcmEventList []eventTrackerEntry
	apiLogEntries []string
)

type eventCheckOpts struct {
	logDir string
	baseList string
	podName string
	podNameHas string
	listtype string
	eventdiff bool
}

func (e eventTrackerEntry) Print(){
	fmt.Printf("%s\t%s\t%s\t%s\t%s\t%s\t %s\n", e.dataPoint, e.eventType, e.namespace, e.objName, e.reflectType, e.resourceVersion, e.timestamp)
	return
}

type eventTrackerEntry struct {
	dataPoint       string
	eventType       string
	namespace       string
	objName         string
	reflectType     string
	resourceVersion string
	trackinfo       string
	uid             string
	timestamp       string
}

func main() {

	flags := eventCheckCmd.Flags()
	flags.StringVar(&ecOpts.logDir, "logDir", "/var/log", "absolute path to the log directory")
	flags.StringVar(&ecOpts.baseList, "baseList", "ETCD", "base list that you want to compare with (choose one from etcd, outetcd, inapiserver, outapiserver, scheduler, controllermanager)")
	flags.StringVar(&ecOpts.podName, "podName", "", "interested pod name")
	flags.StringVar(&ecOpts.podNameHas, "podNameHas", "", "interested in pods whose name contains this string")
	flags.StringVar(&ecOpts.listtype, "listtype", "", "listing event or rv")
	flags.BoolVar(&ecOpts.eventdiff, "eventdiff", false, "listing diff event for each pod")
	eventCheckCmd.Execute()

}


func runEventCheck(){
	fmt.Println("here")
	baseNum := 0

	fmt.Println("Checking logs...")
	err := FilterDirs("kube-apiserver.log", ecOpts.logDir)

	if err != nil {
		log.Fatal("Error reading apiserver log from logDir!")
	}

	err = FilterDirs("kube-scheduler.log", ecOpts.logDir)
	if err != nil {
		log.Fatal("Error reading scheduler log from logDir!")
	}

	err = FilterDirs("kube-controller-manager.log", ecOpts.logDir)
	if err != nil {
		log.Fatal("Error reading scheduler log from logDir!")
	}

	fmt.Printf("length of schedulerEventList %d, kcmeventlist %d\n", len(schedulerEventList), len(kcmEventList))

	podsList := getPodsList(apiLogEntries)
	podCnt := len(podsList)
	fmt.Printf("\n%d pods found\n\n", podCnt)
	falsePodsCnt := 0

	fmt.Printf("\nChecking resourceVersion of event for each pod...\n")
	var falsePodList []string

	for _, podName := range podsList {
		rvLists = make([][]int, 12)
		eventLists = make([][]eventTrackerEntry, 12)
		fillList4APIServer(apiEventList, podName)
		fillList4Client(schedulerEventList, 9, podName)
		fillList4Client(kcmEventList, 10, podName)
		for _, rvlist := range rvLists {
			sort.Ints(rvlist)
		}
		falseList, isSame := compareLists(baseNum, podName)
		if !isSame {
			fmt.Printf("Pod %s is not right in lists: ", podName)
			fmt.Println(falseList)
			fmt.Println("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
			falsePodsCnt ++
			falsePodList = append(falsePodList, podName)
		}
	}

	if falsePodsCnt != 0 {
		fmt.Println("==================================Here are the false pod==================================")
		fmt.Println(falsePodList)
		fmt.Println()

	}
	fmt.Printf("\n%d out of %d pods have problem!\n", falsePodsCnt, podCnt)
}

func FilterDirs(prefix string, dir string) error {
	files, err := ioutil.ReadDir(dir)
	if err != nil {
		fmt.Println(err)
		return err
	}
	for _, f := range files {
		//if prefix == "kube-scheduler.log" || prefix == "kube-controller-manager.log" {
		//	if f.Name() != "2" {
		//		continue
		//	}
		//}
		currdir := filepath.Join(dir, f.Name())
		//fmt.Println(currdir)
		currfiles, currerr := ioutil.ReadDir(currdir)
		if currerr != nil {
			return currerr
		}
		for _, nf := range currfiles {
			if !nf.IsDir() && nf.Name() == prefix{
				currLog := filepath.Join(currdir, nf.Name())
				currEventList, currLogEntries, err := readLogs(currLog)
				if err != nil {
					fmt.Println(err)
					os.Exit(1)
				}
				switch prefix {
				case "kube-apiserver.log":
					apiEventList = append(apiEventList, currEventList...)
					apiLogEntries = append(apiLogEntries, currLogEntries...)
				case "kube-scheduler.log":
					schedulerEventList = append(schedulerEventList, currEventList...)
				case "kube-controller-manager.log":
					kcmEventList = append(kcmEventList, currEventList...)
				}

			}
		}
	}
	return nil
}

func readLogs(logName string) ( []eventTrackerEntry, []string, error) {
	eventLists, logEntries, err := readLines(logName)
	if err != nil {
		log.Fatalf("Failed reading lines: %v", err)
	}
	return eventLists, logEntries, err
}

func readLines(path string) ( []eventTrackerEntry, []string, error) {
	file, err := os.Open(path)
	if err != nil {
		fmt.Println()
		return nil, nil, err
	}
	defer file.Close()

	var lines []string
	var eventEntryList []eventTrackerEntry
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		currLine := scanner.Text()
		if !strings.Contains(currLine, "]"){
			continue
		}
		split1 := strings.Split(currLine, "]")
		if (!strings.HasPrefix(split1[1], " et,")) && (!strings.HasPrefix(split1[1], " et(k8s.io/kubernetes/vendor/k8s.io/client-go/informers/factory.go:87),")) &&
			(!strings.HasPrefix(split1[1], " et(k8s.io/kubernetes/cmd/kube-scheduler/app/server.go:594),")) {
			continue
		}

		result := strings.Split(split1[1], ",")
		if len(result) == 2 {
			//fmt.Println("-------2----------")
			continue
		}
		a := strings.Split(split1[0], " ")
		lines = append(lines, currLine)
		var currEntry eventTrackerEntry
		currEntry.dataPoint = a[len(a) - 1]
		currEntry.eventType = result[1]
		currEntry.namespace = result[2]
		currEntry.objName   = result[3]
		currEntry.reflectType = result[4]
		currEntry.resourceVersion = result[5]
		currEntry.trackinfo = result[6]
		currEntry.uid = result[7]
		currEntry.timestamp = a[1]
		eventEntryList = append(eventEntryList, currEntry)
	}
	return eventEntryList, lines, scanner.Err()
}

func getPodsList(apiLogEntries []string) []string {
	fmt.Printf("\nGetting pods list...")
	var pods []string
	for _, line := range apiLogEntries {
		split1 := strings.Split(line, "]")
		a := strings.Split(split1[0], " ")
		loc := a[len(a) - 1]
		result := strings.Split(split1[1], ",")
		objName := result[3]
		reflectType := result[4]
		//fmt.Printf("wenjia loc is %s, objname is %s, reflectType is %s\n", loc, objName, reflectType)
		if (loc == "watch_cache.go:242") && ( reflectType == "*core.Pod") && strings.HasPrefix(objName, "cust-site"){
			pods = AppendPodIfMissing(pods, objName)
		}
	}
	return pods
}

func fillList4APIServer(entryList []eventTrackerEntry, podName string) {
	for _, entry := range entryList {
		// if  strings.Contains(entry.objName, podName) {
		//result := strings.Split(entry.namespace, "-")
		if  entry.objName == podName {
			rv, _ := strconv.Atoi(entry.resourceVersion)
			switch entry.dataPoint {
			case "watcher.go:303", "watcher.go:293":
				rvLists[0] = AppendIfMissing(rvLists[0], rv)
				eventLists[0] = append(eventLists[0], entry)
			case "watcher.go:252":
				rvLists[1] = AppendIfMissing(rvLists[1], rv)
				eventLists[1] = append(eventLists[1], entry)
			case "watch_cache.go:242":
				rvLists[2] = AppendIfMissing(rvLists[2], rv)
				eventLists[2] = append(eventLists[2], entry)
			case "cacher.go:623":
				rvLists[3] = AppendIfMissing(rvLists[3], rv)
				eventLists[3] = append(eventLists[3], entry)
			case "cacher.go:840":
				rvLists[4] = AppendIfMissing(rvLists[4], rv)
				eventLists[4] = append(eventLists[4], entry)
			case "cacher.go:846", "cacher.go:869", "cacher.go:882":
				rvLists[5] = AppendIfMissing(rvLists[5], rv)
				eventLists[5] = append(eventLists[5], entry)
			case "cacher.go:1009", "cacher.go:981":
				rvLists[6] = AppendIfMissing(rvLists[6], rv)
				eventLists[6] = append(eventLists[6], entry)
			//case "cacher/send1":
			//	rvLists[7] = AppendIfMissing(rvLists[7], rv)
			//	eventLists[7] = append(eventLists[7], entry)
			case "cacher.go:928":
				rvLists[8] = AppendIfMissing(rvLists[8], rv)
				eventLists[8] = append(eventLists[8], entry)
			case "watch.go:235":
				rvLists[11] = AppendIfMissing(rvLists[11], rv)
				eventLists[11] = append(eventLists[11], entry)
			}
		}
	}
	return
}

func fillList4Client(entryList []eventTrackerEntry, listNum int, podName string) {
	//fmt.Printf("wenjia pod name is %s\n", podName)
	for _, entry := range entryList {
		//fmt.Printf("wenjia current entry for list %d\n", listNum)
		//entry.Print()
		//fmt.Printf("entry.dataPoint: (%s), objname: (%s); rv: (%s) \n", entry.dataPoint, entry.objName, entry.resourceVersion)
		//break
		if  entry.objName == podName {
			rv, _ := strconv.Atoi(entry.resourceVersion)
			if strings.HasPrefix(entry.dataPoint, "reflector.go:402") {
				rvLists[listNum] = AppendIfMissing(rvLists[listNum], rv)
				eventLists[listNum] = append(eventLists[listNum], entry)
			}else{
				fmt.Println("WRONG LOG?!")
			}
		}
	}
	return
}

func compareLists(baseNum int, podName string) ([]int, bool) {
	isSame := true
	baseRVList := rvLists[baseNum]
	baseEventList := eventLists[baseNum]
	var falseList []int
	for i := 0; i < 11; i++ {
		if i == baseNum  || i == 7{
			continue
		}
		issame, falservlist := compareWithBase(baseRVList, rvLists[i])
		if !issame {

			fmt.Printf("List %d is DIFFERENT from base list for pod %s\n", i, podName)
			if len(falseList) == 0 {
				falseList = append(falseList, baseNum)
			}
			falseList = append(falseList, i)
			isSame = false
			switch strings.ToUpper(ecOpts.listtype) {
			case "RV":
				if (ecOpts.podName == "all") || (ecOpts.podName == podName) || (ecOpts.podNameHas != "" && strings.Contains(podName, ecOpts.podNameHas)){
					fmt.Printf("false resourceVersions: ")
					fmt.Println(falservlist)
				}
			case "EVENT":
				if (ecOpts.podName == "all") || (ecOpts.podName == podName) || (ecOpts.podNameHas != "" && strings.Contains(podName, ecOpts.podNameHas)){
					fmt.Printf("baselist %d: \n", baseNum)
					for _,entry := range baseEventList {
						entry.Print()
					}

					fmt.Printf("currList %d: \n", i)
					for _,entry := range eventLists[i] {
						entry.Print()
					}
				}
			case "BOTH":
				if (ecOpts.podName == "all") || (ecOpts.podName == podName) || (ecOpts.podNameHas != "" && strings.Contains(podName, ecOpts.podNameHas)){
					fmt.Printf("false resourceVersions: ")
					fmt.Println(falservlist)
					fmt.Printf("baselist %d: \n", baseNum)
					for _,entry := range baseEventList {
						entry.Print()
					}

					fmt.Printf("currList %d: \n", i)
					for _,entry := range eventLists[i] {
						entry.Print()
					}
				}
			}
		}
	}
	return falseList,isSame
}

func compareWithBase(baseRVList []int, currRvList []int) (bool, []int){
	isSame := true
	var falservlist []int
	bl := len(baseRVList)
	cl := len(currRvList)
	if bl != cl {
		fmt.Printf("\nbase rv list has %d events but current rv list has %d events\n", bl, cl)
		isSame = false
	}


	if ecOpts.eventdiff {
		if bl == 0 {
			fmt.Printf("Events missing in base list: \n")
			fmt.Println(currRvList)
			return isSame, currRvList
		}

		if cl == 0 {
			fmt.Printf("Events missing in current list: \n")
			fmt.Println(baseRVList)
			return isSame, baseRVList
		}
	}


	for i, j := 0, 0; i < bl && j < cl; {
		if baseRVList[i] == currRvList[j] {
			i++
			j++
			for (i == bl) && (j < cl) {
				isSame = false
				if ecOpts.eventdiff {
					fmt.Printf("Event %d is missing in base list but exists in current list\n", currRvList[j])
				}
				falservlist = append(falservlist, currRvList[j])
				j++
			}
			for (i < bl) && (j == cl) {
				isSame = false
				if ecOpts.eventdiff {
					fmt.Printf("Event %d is missing in current list but exists in base list\n", baseRVList[i])
				}
				falservlist = append(falservlist, baseRVList[i])
				i++
			}
		} else {
			for baseRVList[i] != currRvList[j] {isSame = false
				if baseRVList[i] < currRvList[j] {
					if ecOpts.eventdiff {
						fmt.Printf("Event %d is missing in current list but exists in base list\n", baseRVList[i])
					}
					falservlist = append(falservlist, baseRVList[i])
					i++
				} else {
					if ecOpts.eventdiff {
						fmt.Printf("Event %d is missing in base list but exists in current list\n", currRvList[j])
					}
					falservlist = append(falservlist, currRvList[j])
					j++
				}
				if i == bl {
					for ;j < cl; j++ {
						if ecOpts.eventdiff {
							fmt.Printf("Event %d is missing in base list but exists in current list\n", currRvList[j])
						}
						falservlist = append(falservlist, currRvList[j])
					}
					return isSame, falservlist
				}
				if j == cl {
					for ;i < bl; i++ {
						if ecOpts.eventdiff {
							fmt.Printf("Event %d is missing in current list but exists in base list\n", baseRVList[i])
						}
						falservlist = append(falservlist, baseRVList[i])
					}
					return isSame, falservlist
				}
			}
		}
	}

	return isSame, falservlist
}


func AppendIfMissing(slice []int, i int) []int {
	for _, ele := range slice {
		if ele == i {
			return slice
		}
	}
	return append(slice, i)
}

func AppendPodIfMissing(slice []string, i string) []string {
	for _, ele := range slice {
		if ele == i {
			return slice
		}
	}
	return append(slice, i)
}

//func AppendEventIfMissing(slice []eventTrackerEntry, rvlist []string , i eventTrackerEntry) []eventTrackerEntry {
//	for _, rv := range rvlist {
//		if rv == i.resourceVersion {
//			return slice
//		}
//	}
//	return append(slice, i)
//}

func listEntry4FalsePod (podName string, apiEventList []eventTrackerEntry){
	var entries []eventTrackerEntry
	for _, entry := range apiEventList {
		if entry.objName == podName {
			entries = append(entries, entry)
			entry.Print()
		}
	}
	return
}

func listRV4FalsePod (podName string, apiEventList []eventTrackerEntry){
	var rvs []string
	for _, entry := range apiEventList {
		if entry.objName == podName {
			rvs = append(rvs, entry.resourceVersion)
		}
	}
	fmt.Println(rvs)
	return
}

func getNewPodNameList (podNameHas string, falsePodList []string) []string {
	var newPodList []string
	for _,falsePod := range falsePodList {
		if strings.Contains(falsePod, podNameHas){
			newPodList = append(newPodList, falsePod)
		}
	}
	return newPodList
}

//func printEvents4FalsePods(falsePodList []string, kasEntryList []eventTrackerEntry, schedulerEntryList []eventTrackerEntry, kcmEntryList []eventTrackerEntry){
//	fmt.Println()
//	if ecOpts.podName != "" {
//		fmt.Printf("\n==================================List events for pod %s==================================\n", ecOpts.podName)
//		fmt.Println("\nEvent in apiserver: ")
//		listEntry4FalsePod(ecOpts.podName, kasEntryList)
//		fmt.Println("\nEvent in scheduler: ")
//		listEntry4FalsePod(ecOpts.podName, schedulerEntryList)
//		fmt.Println("\nEvent in controller-manager: ")
//		listEntry4FalsePod(ecOpts.podName, kcmEntryList)
//	} else if ecOpts.podName == "all"{
//		for _, podName := range falsePodList {
//			fmt.Printf("\n==================================List events for pod %s==================================\n", podName)
//			fmt.Println("\nEvent in apiserver: ")
//			listEntry4FalsePod(podName, kasEntryList)
//			fmt.Println("\nEvent in scheduler: ")
//			listEntry4FalsePod(podName, schedulerEntryList)
//			fmt.Println("\nEvent in controller-manager: ")
//			listEntry4FalsePod(podName, kcmEntryList)
//		}
//	} else if ecOpts.podNameHas != "" {
//		newPodLists := getNewPodNameList(ecOpts.podNameHas, falsePodList)
//		for _, podName := range newPodLists {
//			fmt.Printf("\n==================================List events for pod %s==================================\n", podName)
//			fmt.Println("\nEvent in apiserver: ")
//			listEntry4FalsePod(podName, kasEntryList)
//			fmt.Println("\nEvent in scheduler: ")
//			listEntry4FalsePod(podName, schedulerEntryList)
//			fmt.Println("\nEvent in controller-manager: ")
//			listEntry4FalsePod(podName, kcmEntryList)
//		}
//	}
//}
//
//func printRV4FalsePods(falsePodList []string, kasEntryList []eventTrackerEntry, schedulerEntryList []eventTrackerEntry, kcmEntryList []eventTrackerEntry){
//	fmt.Println()
//
//	if ecOpts.podName != "" {
//		fmt.Printf("\n==================================List RV for pod %s==================================\n", ecOpts.podName)
//		fmt.Println("\nEvent in apiserver: ")
//		listRV4FalsePod(ecOpts.podName, kasEntryList)
//		fmt.Println("\nEvent in scheduler: ")
//		listRV4FalsePod(ecOpts.podName, schedulerEntryList)
//		fmt.Println("\nEvent in controller-manager: ")
//		listRV4FalsePod(ecOpts.podName, kcmEntryList)
//	} else if ecOpts.podName == "all"{
//		for _, podName := range falsePodList {
//			fmt.Printf("\n==================================List events for pod %s==================================\n", podName)
//			fmt.Println("\nEvent in apiserver: ")
//			listRV4FalsePod(podName, kasEntryList)
//			fmt.Println("\nEvent in scheduler: ")
//			listRV4FalsePod(podName, schedulerEntryList)
//			fmt.Println("\nEvent in controller-manager: ")
//			listRV4FalsePod(podName, kcmEntryList)
//		}
//	} else if ecOpts.podNameHas != "" {
//		newPodLists := getNewPodNameList(ecOpts.podNameHas, falsePodList)
//		for _, podName := range newPodLists {
//			fmt.Printf("\n==================================List events for pod %s==================================\n", podName)
//			fmt.Println("\nEvent in apiserver: ")
//			listRV4FalsePod(podName, kasEntryList)
//			fmt.Println("\nEvent in scheduler: ")
//			listRV4FalsePod(podName, schedulerEntryList)
//			fmt.Println("\nEvent in controller-manager: ")
//			listRV4FalsePod(podName, kcmEntryList)
//		}
//	}
//}


