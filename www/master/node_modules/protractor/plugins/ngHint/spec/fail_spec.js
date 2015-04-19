describe('check if ngHint plugin works on bad apps', function() {
  it('should have ngHint problems on bad apps', function() {
    browser.get('ngHint/noNgHint.html');
    browser.get('ngHint/noTag.html');
    browser.get('ngHint/unused.html');
  });
});
