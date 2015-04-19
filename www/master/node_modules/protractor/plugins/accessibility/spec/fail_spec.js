describe('check if accessibility plugin works on bad apps', function() {
  it('should have accessibility problems on markup', function() {
    browser.get('accessibility/badMarkup.html');
  });
});
