// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux

// Package main contains a program that displays the IPv4 address
// of the machine on an a Grove-LCD RGB backlight.
package main

import (
	"fmt"
	"net"

	"golang.org/x/exp/io/i2c"
)

const (
	DISPLAY_RGB_ADDR  = 0x62
	DISPLAY_TEXT_ADDR = 0x3e
)

func main() {
	d, err := i2c.Open(&i2c.Devfs{Dev: "/dev/i2c-1", Addr: DISPLAY_RGB_ADDR})
	if err != nil {
		panic(err)
	}

	td, err := i2c.Open(&i2c.Devfs{Dev: "/dev/i2c-1", Addr: DISPLAY_TEXT_ADDR})
	if err != nil {
		panic(err)
	}

	// Set the backlight color to 100,100,100.
	write(d, []byte{0, 0})
	write(d, []byte{1, 0})
	write(d, []byte{0x08, 0xaa})
	write(d, []byte{4, 100}) // R value
	write(d, []byte{3, 100}) // G value
	write(d, []byte{2, 100}) // B value

	ip, err := resolveIP()
	if err != nil {
		panic(err)
	}

	fmt.Printf("host machine IP is %v\n", ip)

	write(td, []byte{0x80, 0x02})        // return home
	write(td, []byte{0x80, 0x01})        // clean the display
	write(td, []byte{0x80, 0x08 | 0x04}) // no cursor
	write(td, []byte{0x80, 0x28})        // two lines

	for _, s := range ip {
		write(td, []byte{0x40, byte(s)})
	}
}

func write(d *i2c.Device, buf []byte) {
	err := d.Write(buf)
	if err != nil {
		panic(err)
	}
}

func resolveIP() (string, error) {
	var ip net.IP
	ifaces, err := net.Interfaces()
	if err != nil {
		panic(err)
	}
	for _, i := range ifaces {
		addrs, err := i.Addrs()
		if err != nil {
			panic(err)
		}
		for _, addr := range addrs {
			switch v := addr.(type) {
			case *net.IPNet:
				ip = v.IP
			case *net.IPAddr:
				ip = v.IP
			}
			ip = ip.To4()
			if ip != nil && ip.String() != "127.0.0.1" {
				return ip.String(), nil
			}
		}
	}
	return "", fmt.Errorf("cannot resolve the IP")
}
