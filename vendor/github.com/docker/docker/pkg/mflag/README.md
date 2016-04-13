Package mflag (aka multiple-flag) implements command-line flag parsing.  
It's an **hacky** fork of the [official golang package](http://golang.org/pkg/flag/)

It adds:

* both short and long flag version  
`./example -s red` `./example --string blue`

* multiple names for the same option  
```
$>./example -h
Usage of example:
  -s, --string="": a simple string
```

___
It is very flexible on purpose, so you can do things like:  
```
$>./example -h
Usage of example:
  -s, -string, --string="": a simple string
```

Or:  
```
$>./example -h
Usage of example:
  -oldflag, --newflag="": a simple string
```

You can also hide some flags from the usage, so if we want only `--newflag`:  
```
$>./example -h
Usage of example:
  --newflag="": a simple string
$>./example -oldflag str
str
```

See [example.go](example/example.go) for more details.
