describe("less.js production behaviour", function() {
    it("doesn't log any messages", function() {
        expect(logMessages.length).toEqual(0);
    });
});
