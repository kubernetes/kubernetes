netdata -W buildinfo
# If get "netdata: command not found", try (required running Netdata)
$(ps aux | grep -m1 -E -o "[a-zA-Z/]+netdata ") -W buildinfo
