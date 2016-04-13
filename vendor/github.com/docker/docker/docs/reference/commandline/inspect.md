<!--[metadata]>
+++
title = "inspect"
description = "The inspect command description and usage"
keywords = ["inspect, container, json"]
[menu.main]
parent = "smn_cli"
weight=1
+++
<![end-metadata]-->

# inspect

    Usage: docker inspect [OPTIONS] CONTAINER|IMAGE [CONTAINER|IMAGE...]

    Return low-level information on a container or image

      -f, --format=""    Format the output using the given go template

     --type=container|image  Return JSON for specified type, permissible 
                             values are "image" or "container"

By default, this will render all results in a JSON array. If a format is
specified, the given template will be executed for each result.

Go's [text/template](http://golang.org/pkg/text/template/) package
describes all the details of the format.

## Examples

**Get an instance's IP address:**

For the most part, you can pick out any field from the JSON in a fairly
straightforward manner.

    $ docker inspect --format='{{.NetworkSettings.IPAddress}}' $INSTANCE_ID

**Get an instance's MAC Address:**

For the most part, you can pick out any field from the JSON in a fairly
straightforward manner.

    $ docker inspect --format='{{.NetworkSettings.MacAddress}}' $INSTANCE_ID

**Get an instance's log path:**

    $ docker inspect --format='{{.LogPath}}' $INSTANCE_ID

**List All Port Bindings:**

One can loop over arrays and maps in the results to produce simple text
output:

    $ docker inspect --format='{{range $p, $conf := .NetworkSettings.Ports}} {{$p}} -> {{(index $conf 0).HostPort}} {{end}}' $INSTANCE_ID

**Find a Specific Port Mapping:**

The `.Field` syntax doesn't work when the field name begins with a
number, but the template language's `index` function does. The
`.NetworkSettings.Ports` section contains a map of the internal port
mappings to a list of external address/port objects, so to grab just the
numeric public port, you use `index` to find the specific port map, and
then `index` 0 contains the first object inside of that. Then we ask for
the `HostPort` field to get the public address.

    $ docker inspect --format='{{(index (index .NetworkSettings.Ports "8787/tcp") 0).HostPort}}' $INSTANCE_ID

**Get config:**

The `.Field` syntax doesn't work when the field contains JSON data, but
the template language's custom `json` function does. The `.config`
section contains complex JSON object, so to grab it as JSON, you use
`json` to convert the configuration object into JSON.

    $ docker inspect --format='{{json .config}}' $INSTANCE_ID

