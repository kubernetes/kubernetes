#!/usr/bin/perl
#
# A simple helper script to help people build seccomp profiles for
# Docker/LXC.  The goal is mostly to reduce the attack surface to the
# kernel, by restricting access to rarely used, recently added or not used
# syscalls.
#
# This script processes one or more files which contain the list of system
# calls to be allowed.  See mkseccomp.sample for more information how you
# can configure the list of syscalls.  When run, this script produces output
# which, when stored in a file, can be passed to docker as follows:
#
# docker run --lxc-conf="lxc.seccomp=$file" <rest of arguments>
#
# The included sample file shows how to cut about a quarter of all syscalls,
# which affecting most applications.
#
# For specific situations it is possible to reduce the list further. By
# reducing the list to just those syscalls required by a certain application
# you can make it difficult for unknown/unexpected code to run.
#
# Run this script as follows:
#
# ./mkseccomp.pl < mkseccomp.sample >syscalls.list
# or
# ./mkseccomp.pl mkseccomp.sample >syscalls.list
#
# Multiple files can be specified, in which case the lists of syscalls are
# combined.
#
# By Martijn van Oosterhout <kleptog@svana.org> Nov 2013

# How it works:
#
# This program basically spawns two processes to form a chain like:
#
# <process data section to prefix __NR_> | cpp | <add header and filter unknown syscalls>

use strict;
use warnings;

if( -t ) {
    print STDERR "Helper script to make seccomp filters for Docker/LXC.\n";
    print STDERR "Usage: mkseccomp.pl < [files...]\n";
    exit 1;
}

my $pid = open(my $in, "-|") // die "Couldn't fork1 ($!)\n";

if($pid == 0) {  # Child
    $pid = open(my $out, "|-") // die "Couldn't fork2 ($!)\n";

    if($pid == 0) { # Child, which execs cpp
        exec "cpp" or die "Couldn't exec cpp ($!)\n";
        exit 1;
    }

    # Process the DATA section and output to cpp
    print $out "#include <sys/syscall.h>\n";
    while(<>) {
        if(/^\w/) {
            print $out "__NR_$_";
        }
    }
    close $out;
    exit 0;

}

# Print header and then process output from cpp.
print "1\n";
print "whitelist\n";

while(<$in>) {
    print if( /^[0-9]/ );
}

