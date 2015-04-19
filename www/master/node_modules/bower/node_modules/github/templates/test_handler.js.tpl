    it("should successfully execute <%name%>",  function(next) {
        client.<%funcName%>(
            <%params%>,
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });