# NAME
   runc list - lists containers started by runc with the given root

# SYNOPSIS
   runc list [command options]

# EXAMPLE
Where the given root is specified via the global option "--root"
(default: "/run/runc").

To list containers created via the default "--root":
       # runc list

To list containers created using a non-default value for "--root":
       # runc --root value list

# OPTIONS
   --format value, -f value     select one of: table or json (default: "table")
   --quiet, -q                  display only container IDs
