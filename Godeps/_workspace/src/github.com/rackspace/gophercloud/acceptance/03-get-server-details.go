// +build acceptance,old

package main

import (
	"flag"
	"fmt"
	"github.com/rackspace/gophercloud"
	"os"
	"github.com/racker/perigee"
)

var id = flag.String("i", "", "Server ID to get info on.  Defaults to first server in your account if unspecified.")
var rgn = flag.String("r", "", "Datacenter region.  Leave blank for default region.")
var quiet = flag.Bool("quiet", false, "Run quietly, for acceptance testing.  $? non-zero if issue.")

func main() {
	flag.Parse()

	resultCode := 0
	withIdentity(false, func(auth gophercloud.AccessProvider) {
		withServerApi(auth, func(servers gophercloud.CloudServersProvider) {
			var (
				err              error
				serverId         string
				deleteAfterwards bool
			)

			// Figure out which server to provide server details for.
			if *id == "" {
				deleteAfterwards, serverId, err = locateAServer(servers)
				if err != nil {
					panic(err)
				}
				if deleteAfterwards {
					defer servers.DeleteServerById(serverId)
				}
			} else {
				serverId = *id
			}

			// Grab server details by ID, and provide a report.
			s, err := servers.ServerById(serverId)
			if err != nil {
				panic(err)
			}

			configs := []string{
				"Access IPv4: %s\n",
				"Access IPv6: %s\n",
				"    Created: %s\n",
				"     Flavor: %s\n",
				"    Host ID: %s\n",
				"         ID: %s\n",
				"      Image: %s\n",
				"       Name: %s\n",
				"   Progress: %s\n",
				"     Status: %s\n",
				"  Tenant ID: %s\n",
				"    Updated: %s\n",
				"    User ID: %s\n",
			}

			values := []string{
				s.AccessIPv4,
				s.AccessIPv6,
				s.Created,
				s.Flavor.Id,
				s.HostId,
				s.Id,
				s.Image.Id,
				s.Name,
				fmt.Sprintf("%d", s.Progress),
				s.Status,
				s.TenantId,
				s.Updated,
				s.UserId,
			}

			if !*quiet {
				fmt.Println("Server info:")
				for i, _ := range configs {
					fmt.Printf(configs[i], values[i])
				}
			}
		})

		// Negative test -- We should absolutely never panic for a server that doesn't exist.
		withServerApi(auth, func(servers gophercloud.CloudServersProvider) {
			_, err := servers.ServerById(randomString("garbage", 32))
			if err == nil {
				fmt.Printf("Expected a 404 response when looking for a server known not to exist\n")
				resultCode = 1
			}
			perigeeError, ok := err.(*perigee.UnexpectedResponseCodeError)
			if !ok {
				fmt.Printf("Unexpected error type\n")
				resultCode = 1
			} else {
				if perigeeError.Actual != 404 {
					fmt.Printf("Expected a 404 error code\n")
				}
			}
		})
	})
	os.Exit(resultCode)
}

// locateAServer queries the set of servers owned by the user.  If at least one
// exists, the first found is picked, and its ID is returned.  Otherwise, a new
// server will be created, and its ID returned.
//
// deleteAfter will be true if the caller should schedule a call to DeleteServerById()
// to clean up.
func locateAServer(servers gophercloud.CloudServersProvider) (deleteAfter bool, id string, err error) {
	ss, err := servers.ListServers()
	if err != nil {
		return false, "", err
	}

	if len(ss) > 0 {
		// We could just cheat and dump the server details from ss[0].
		// But, that tests ListServers(), and not ServerById().  So, we
		// elect not to cheat.
		return false, ss[0].Id, nil
	}

	serverId, err := createServer(servers, "", "", "", "")
	if err != nil {
		return false, "", err
	}
	err = waitForServerState(servers, serverId, "ACTIVE")
	return true, serverId, err
}
