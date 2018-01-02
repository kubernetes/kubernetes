package dhcp4client_test

import (
	"log"
	"net"
	"testing"

	"github.com/d2g/dhcp4client"
)

//Example Client
func Test_ExampleClient(test *testing.T) {
	var err error

	m, err := net.ParseMAC("08-00-27-00-A8-E8")
	if err != nil {
		log.Printf("MAC Error:%v\n", err)
	}

	//Create a connection to use
	//We need to set the connection ports to 1068 and 1067 so we don't need root access
	c, err := dhcp4client.NewInetSock(dhcp4client.SetLocalAddr(net.UDPAddr{IP: net.IPv4(0, 0, 0, 0), Port: 1068}), dhcp4client.SetRemoteAddr(net.UDPAddr{IP: net.IPv4bcast, Port: 1067}))
	if err != nil {
		test.Error("Client Connection Generation:" + err.Error())
	}
	defer c.Close()

	exampleClient, err := dhcp4client.New(dhcp4client.HardwareAddr(m), dhcp4client.Connection(c))
	if err != nil {
		test.Fatalf("Error:%v\n", err)
	}
	defer exampleClient.Close()

	success, acknowledgementpacket, err := exampleClient.Request()

	test.Logf("Success:%v\n", success)
	test.Logf("Packet:%v\n", acknowledgementpacket)

	if err != nil {
		networkError, ok := err.(*net.OpError)
		if ok && networkError.Timeout() {
			test.Log("Test Skipping as it didn't find a DHCP Server")
			test.SkipNow()
		}
		test.Fatalf("Error:%v\n", err)
	}

	if !success {
		test.Error("We didn't sucessfully get a DHCP Lease?")
	} else {
		log.Printf("IP Received:%v\n", acknowledgementpacket.YIAddr().String())
	}

	test.Log("Start Renewing Lease")
	success, acknowledgementpacket, err = exampleClient.Renew(acknowledgementpacket)
	if err != nil {
		networkError, ok := err.(*net.OpError)
		if ok && networkError.Timeout() {
			test.Log("Renewal Failed! Because it didn't find the DHCP server very Strange")
			test.Errorf("Error" + err.Error())
		}
		test.Fatalf("Error:%v\n", err)
	}

	if !success {
		test.Error("We didn't sucessfully Renew a DHCP Lease?")
	} else {
		log.Printf("IP Received:%v\n", acknowledgementpacket.YIAddr().String())
	}

}

//Example Client (With MathGID)
func Test_ExampleClientWithMathGenerateXID(test *testing.T) {
	var err error

	m, err := net.ParseMAC("08-00-27-00-A8-E8")
	if err != nil {
		log.Printf("MAC Error:%v\n", err)
	}

	//Create a connection to use
	//We need to set the connection ports to 1068 and 1067 so we don't need root access
	c, err := dhcp4client.NewInetSock(dhcp4client.SetLocalAddr(net.UDPAddr{IP: net.IPv4(0, 0, 0, 0), Port: 1068}), dhcp4client.SetRemoteAddr(net.UDPAddr{IP: net.IPv4bcast, Port: 1067}))
	if err != nil {
		test.Error("Client Connection Generation:" + err.Error())
	}
	defer c.Close()

	// If you ar using MathGenerateXID then you are responsible for seeding math/rand
	exampleClient, err := dhcp4client.New(dhcp4client.HardwareAddr(m), dhcp4client.Connection(c), dhcp4client.GenerateXID(dhcp4client.MathGenerateXID))
	if err != nil {
		test.Fatalf("Error:%v\n", err)
	}
	defer exampleClient.Close()

	success, acknowledgementpacket, err := exampleClient.Request()

	test.Logf("Success:%v\n", success)
	test.Logf("Packet:%v\n", acknowledgementpacket)

	if err != nil {
		networkError, ok := err.(*net.OpError)
		if ok && networkError.Timeout() {
			test.Log("Test Skipping as it didn't find a DHCP Server")
			test.SkipNow()
		}
		test.Fatalf("Error:%v\n", err)
	}

	if !success {
		test.Error("We didn't sucessfully get a DHCP Lease?")
	} else {
		log.Printf("IP Received:%v\n", acknowledgementpacket.YIAddr().String())
	}
}
