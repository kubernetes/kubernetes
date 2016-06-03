# Nethealth HTTP Check

 The nethealth binary performs a quick HTTP GET download speed check
 Key Features:
 - Shell-script friendly - returns a non-zero exit code on timeout or corruption or bandwidth threshold failures
 - Timeout configurable to abort the test early (for super slow links)
 - Can compare actual bandwidth against a command line minimum bandwidth parameter and return a non-zero exit code.
 - Corruption check - can download a checksum file and compute blob checksum and compare.
 - Configurable object URL for non-GCE environments
 
 ## Generating content files
 
The following steps can be used if you wish to generate content files for different sizes.

Assumptions: openssl installed and 64MB blob

```
$ openssl rand -out 64MB.bin 67108864
$ openssl sha512 64MB.bin > sha512.txt
$ cat sha512.txt
SHA512(64MB.bin)= 9abddc26a6bda88864326bdc8130f0653562669fcc7f617b26a53ea73781d5cb2eb9418fde53a5743c86aa2c493aeb8933d829a812d275c2fc4ecd84427999bf
$
```


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/util/nethealth/README.md?pixel)]()


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/pkg/util/nethealth/README.md?pixel)]()
