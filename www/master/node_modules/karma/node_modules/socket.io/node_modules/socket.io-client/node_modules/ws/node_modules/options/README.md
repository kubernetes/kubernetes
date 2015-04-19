# options.js #

A very light-weight in-code option parsers for node.js.

## Usage ##

``` js
var Options = require("options");

// Create an Options object
function foo(options) {
        var default_options = {
                foo : "bar"
        };
        
        // Create an option object with default value
        var opts = new Options(default_options);
        
        // Merge options
        opts = opts.merge(options);
        
        // Reset to default value
        opts.reset();
        
        // Copy selected attributes out
        var seled_att = opts.copy("foo");
        
        // Read json options from a file. 
        opts.read("options.file"); // Sync
        opts.read("options.file", function(err){ // Async
                if(err){ // If error occurs
                        console.log("File error.");
                }else{
                        // No error
                }
        });
        
        // Attributes defined or not
        opts.isDefinedAndNonNull("foobar");
        opts.isDefined("foobar");
}

```


## License ##

(The MIT License)

Copyright (c) 2012 Einar Otto Stangvik &lt;einaros@gmail.com&gt;

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
'Software'), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
