% DOCKER(1) Docker User Manuals
% Docker Community
% JUNE 2015
# NAME
docker-version - Show the Docker version information.

# SYNOPSIS
**docker version**
[**--help**]
[**-f**|**--format**[=*FORMAT*]]

# DESCRIPTION
This command displays version information for both the Docker client and 
daemon. 

# OPTIONS
**--help**
    Print usage statement

**-f**, **--format**=""
    Format the output using the given go template.

# EXAMPLES

## Display Docker version information

The default output:

    $ docker version
	Client:
	 Version:      1.8.0
	 API version:  1.20
	 Go version:   go1.4.2
	 Git commit:   f5bae0a
	 Built:        Tue Jun 23 17:56:00 UTC 2015
	 OS/Arch:      linux/amd64

	Server:
	 Version:      1.8.0
	 API version:  1.20
	 Go version:   go1.4.2
	 Git commit:   f5bae0a
	 Built:        Tue Jun 23 17:56:00 UTC 2015
	 OS/Arch:      linux/amd64

Get server version:

    $ docker version --format '{{.Server.Version}}'
	1.8.0

Dump raw data:

To view all available fields, you can use the format `{{json .}}`.

    $ docker version --format '{{json .}}'
    {"Client":{"Version":"1.8.0","ApiVersion":"1.20","GitCommit":"f5bae0a","GoVersion":"go1.4.2","Os":"linux","Arch":"amd64","BuildTime":"Tue Jun 23 17:56:00 UTC 2015"},"ServerOK":true,"Server":{"Version":"1.8.0","ApiVersion":"1.20","GitCommit":"f5bae0a","GoVersion":"go1.4.2","Os":"linux","Arch":"amd64","KernelVersion":"3.13.2-gentoo","BuildTime":"Tue Jun 23 17:56:00 UTC 2015"}}

	
# HISTORY
June 2014, updated by Sven Dowideit <SvenDowideit@home.org.au>
June 2015, updated by John Howard <jhoward@microsoft.com>
June 2015, updated by Patrick Hemmer <patrick.hemmer@gmail.com
