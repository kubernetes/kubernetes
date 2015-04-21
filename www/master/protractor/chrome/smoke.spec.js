describe('Kubernetes UI Chome', function() {
  it('should have all the expected tabs loaded', function() {
    browser.get('http://localhost:8000');
    expect(browser.getTitle()).toEqual('Kubernetes UI');

    // Navigate to the graph page.
    var dashboardTab = element(by.id('tab_001'));
    expect(dashboardTab).toBeDefined();
    dashboardTab.click();
    expect(browser.getLocationAbsUrl()).toBe('/dashboard/');

    // var graphTab = element(by.id('tab_002'));
    // expect(graphTab).toBeDefined();
    // graphTab.click();
    // expect(browser.getLocationAbsUrl()).toBe('/graph/');
  });
});
