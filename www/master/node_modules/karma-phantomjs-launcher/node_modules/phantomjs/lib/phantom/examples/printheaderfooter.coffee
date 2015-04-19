someCallback = (pageNum, numPages) ->
  "<h1> someCallback: " + pageNum + " / " + numPages + "</h1>"
page = require("webpage").create()
system = require("system")
if system.args.length < 3
  console.log "Usage: printheaderfooter.js URL filename"
  phantom.exit 1
else
  address = system.args[1]
  output = system.args[2]
  page.viewportSize =
    width: 600
    height: 600

  page.paperSize =
    format: "A4"
    margin: "1cm"
    
    # default header/footer for pages that don't have custom overwrites (see below) 
    header:
      height: "1cm"
      contents: phantom.callback((pageNum, numPages) ->
        return ""  if pageNum is 1
        "<h1>Header <span style='float:right'>" + pageNum + " / " + numPages + "</span></h1>"
      )

    footer:
      height: "1cm"
      contents: phantom.callback((pageNum, numPages) ->
        return ""  if pageNum is numPages
        "<h1>Footer <span style='float:right'>" + pageNum + " / " + numPages + "</span></h1>"
      )

  page.open address, (status) ->
    if status isnt "success"
      console.log "Unable to load the address!"
    else
      
      # check whether the loaded page overwrites the header/footer setting,
      #               i.e. whether a PhantomJSPriting object exists. Use that then instead
      #               of our defaults above.
      #
      #               example:
      #               <html>
      #                 <head>
      #                   <script type="text/javascript">
      #                     var PhantomJSPrinting = {
      #                        header: {
      #                            height: "1cm",
      #                            contents: function(pageNum, numPages) { return pageNum + "/" + numPages; }
      #                        },
      #                        footer: {
      #                            height: "1cm",
      #                            contents: function(pageNum, numPages) { return pageNum + "/" + numPages; }
      #                        }
      #                     };
      #                   </script>
      #                 </head>
      #                 <body><h1>asdfadsf</h1><p>asdfadsfycvx</p></body>
      #              </html>
      #            
      if page.evaluate(->
        typeof PhantomJSPrinting is "object"
      )
        paperSize = page.paperSize
        paperSize.header.height = page.evaluate(->
          PhantomJSPrinting.header.height
        )
        paperSize.header.contents = phantom.callback((pageNum, numPages) ->
          page.evaluate ((pageNum, numPages) ->
            PhantomJSPrinting.header.contents pageNum, numPages
          ), pageNum, numPages
        )
        paperSize.footer.height = page.evaluate(->
          PhantomJSPrinting.footer.height
        )
        paperSize.footer.contents = phantom.callback((pageNum, numPages) ->
          page.evaluate ((pageNum, numPages) ->
            PhantomJSPrinting.footer.contents pageNum, numPages
          ), pageNum, numPages
        )
        page.paperSize = paperSize
        console.log page.paperSize.header.height
        console.log page.paperSize.footer.height
      window.setTimeout (->
        page.render output
        phantom.exit()
      ), 200
