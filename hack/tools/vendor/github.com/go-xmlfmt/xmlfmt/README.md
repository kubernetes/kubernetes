# Go XML Formatter

[![MIT License](http://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Go Doc](https://img.shields.io/badge/godoc-reference-4b68a3.svg)](https://godoc.org/github.com/go-xmlfmt/xmlfmt)
[![Go Report Card](https://goreportcard.com/badge/github.com/go-xmlfmt/xmlfmt)](https://goreportcard.com/report/github.com/go-xmlfmt/xmlfmt)
[![Codeship Status](https://codeship.com/projects/c49f02b0-a384-0134-fb20-2e0351080565/status?branch=master)](https://codeship.com/projects/190297)

## Synopsis

The Go XML Formatter, xmlfmt, will format the XML string in a readable way. 

```go
package main

import "github.com/go-xmlfmt/xmlfmt"

func main() {
	xml1 := `<root><this><is>a</is><test /><message><org><cn>Some org-or-other</cn><ph>Wouldnt you like to know</ph></org><contact><fn>Pat</fn><ln>Califia</ln></contact></message></this></root>`
	x := xmlfmt.FormatXML(xml1, "\t", "  ")
	print(x)
}

```

Output:

```xml
	<root>
	  <this>
	    <is>a
	    </is>
	    <test />
	    <message>
	      <!-- with comment -->
	      <org>
	        <cn>Some org-or-other
	        </cn>
	        <ph>Wouldnt you like to know
	        </ph>
	      </org>
	      <contact>
	        <fn>Pat
	        </fn>
	        <ln>Califia
	        </ln>
	      </contact>
	    </message>
	  </this>
	</root>
```

There is no XML decoding and encoding involved, only pure regular expression matching and replacing. So it is much faster than going through decoding and encoding procedures. Moreover, the exact XML source string is preserved, instead of being changed by the encoder. This is why this package exists in the first place. 

## Command

To use it on command line, check out [xmlfmt](https://github.com/AntonioSun/xmlfmt):


```
$ xmlfmt 
XML Formatter
built on 2019-12-08

The xmlfmt will format the XML string without rewriting the document

Options:

  -h, --help          display help information
  -f, --file         *The xml file to read from (or stdin)
  -p, --prefix        each element begins on a new line and this prefix
  -i, --indent[=  ]   indent string for nested elements
```


## Justification

### The format

The Go XML Formatter is not called XML Beautifier because the result is not *exactly* as what people would expect -- some, but not all, closing tags stays on the same line, just as shown above. Having been looking at the result and thinking over it, I now think it is actually a better way to present it, as those closing tags on the same line are better stay that way in my opinion. I.e., 

When it comes to very big XML strings, which is what I’m dealing every day, saving spaces by not allowing those closing tags taking extra lines is plus instead of negative to me. 

### The alternative

To format it “properly”, i.e., as what people would normally see, is very hard using pure regular expression. In fact, according to Sam Whited from the go-nuts mlist, 

> Regular expression is, well, regular. This means that they can parse regular grammars, but can't parse context free grammars (like XML). It is actually impossible to use a regex to do this task; it will always be fragile, unfortunately.

So if the output format is so important to you, then unfortunately you have to go through decoding and encoding procedures. But there are some drawbacks as well, as put by James McGill, in http://stackoverflow.com/questions/21117161, besides such method being slow:

> I like this solution, but am still in search of a Golang XML formatter/prettyprinter that doesn't rewrite the document (other than formatting whitespace). Marshalling or using the Encoder will change namespace declarations.
> 
> For example an element like "< ns1:Element />" will be translated to something like '< Element xmlns="http://bla...bla/ns1" >< /Element >' which seems harmless enough except when the intent is to not alter the xml other than formatting. -- James McGill Nov 12 '15

Using Sam's code as an example, 

https://play.golang.org/p/JUqQY3WpW5

The above code formats the following XML

```xml
<soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/"
  xmlns:ns="http://example.com/ns">
   <soapenv:Header/>
   <soapenv:Body>
     <ns:request>
      <ns:customer>
       <ns:id>123</ns:id>
       <ns:name type="NCHZ">John Brown</ns:name>
      </ns:customer>
     </ns:request>
   </soapenv:Body>
</soapenv:Envelope>
```

into this:

```xml
<Envelope xmlns="http://schemas.xmlsoap.org/soap/envelope/" xmlns:_xmlns="xmlns" _xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" _xmlns:ns="http://example.com/ns">
 <Header xmlns="http://schemas.xmlsoap.org/soap/envelope/"></Header>
 <Body xmlns="http://schemas.xmlsoap.org/soap/envelope/">
  <request xmlns="http://example.com/ns">
   <customer xmlns="http://example.com/ns">
    <id xmlns="http://example.com/ns">123</id>
    <name xmlns="http://example.com/ns" type="NCHZ">John Brown</name>
   </customer>
  </request>
 </Body>
</Envelope>
```

I know they are syntactically the same, however the problem is that they *look* totally different.

That's why there is this package, an XML Beautifier that doesn't rewrite the document. 

## Credit

The credit goes to **diotalevi** from his post at http://www.perlmonks.org/?node_id=261292.

However, it does not work for all cases. For example,

```sh
$ echo '<Envelope xmlns=http://schemas.xmlsoap.org/soap/envelope/ xmlns:_xmlns=xmlns _xmlns:soapenv=http://schemas.xmlsoap.org/soap/envelope/ _xmlns:ns=http://example.com/ns><Header xmlns=http://schemas.xmlsoap.org/soap/envelope/></Header><Body xmlns=http://schemas.xmlsoap.org/soap/envelope/><request xmlns=http://example.com/ns><customer xmlns=http://example.com/ns><id xmlns=http://example.com/ns>123</id><name xmlns=http://example.com/ns type=NCHZ>John Brown</name></customer></request></Body></Envelope>' | perl -pe 's/(?<=>)\s+(?=<)//g; s(<(/?)([^/>]+)(/?)>\s*(?=(</?))?)($indent+=$3?0:$1?-1:1;"<$1$2$3>".($1&&($4 eq"</")?"\n".("  "x$indent):$4?"\n".("  "x$indent):""))ge'
```
```xml
<Envelope xmlns=http://schemas.xmlsoap.org/soap/envelope/ xmlns:_xmlns=xmlns _xmlns:soapenv=http://schemas.xmlsoap.org/soap/envelope/ _xmlns:ns=http://example.com/ns><Header xmlns=http://schemas.xmlsoap.org/soap/envelope/></Header>
<Body xmlns=http://schemas.xmlsoap.org/soap/envelope/><request xmlns=http://example.com/ns><customer xmlns=http://example.com/ns><id xmlns=http://example.com/ns>123</id>
<name xmlns=http://example.com/ns type=NCHZ>John Brown</name>
</customer>
</request>
</Body>
</Envelope>
```

I simplified the algorithm, and now it should work for all cases:

```sh
echo '<Envelope xmlns="http://schemas.xmlsoap.org/soap/envelope/" xmlns:_xmlns="xmlns" _xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" _xmlns:ns="http://example.com/ns"><Header xmlns="http://schemas.xmlsoap.org/soap/envelope/"></Header><Body xmlns="http://schemas.xmlsoap.org/soap/envelope/"><request xmlns="http://example.com/ns"><customer xmlns="http://example.com/ns"><id xmlns="http://example.com/ns">123</id><name xmlns="http://example.com/ns" type="NCHZ">John Brown</name></customer></request></Body></Envelope>' | perl -pe 's/(?<=>)\s+(?=<)//g; s(<(/?)([^>]+)(/?)>)($indent+=$3?0:$1?-1:1;"<$1$2$3>"."\n".("  "x$indent))ge'
```
```xml
<Envelope xmlns="http://schemas.xmlsoap.org/soap/envelope/" xmlns:_xmlns="xmlns" _xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" _xmlns:ns="http://example.com/ns">
  <Header xmlns="http://schemas.xmlsoap.org/soap/envelope/">
    </Header>
  <Body xmlns="http://schemas.xmlsoap.org/soap/envelope/">
    <request xmlns="http://example.com/ns">
      <customer xmlns="http://example.com/ns">
        <id xmlns="http://example.com/ns">
          123</id>
        <name xmlns="http://example.com/ns" type="NCHZ">
          John Brown</name>
        </customer>
      </request>
    </Body>
  </Envelope>
```

This package is a direct translate from above Perl code into Go,
then further enhanced by @ruandao.
