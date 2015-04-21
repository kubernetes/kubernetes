describe('Kubernetes UI Dashboard', function() {
  it('should have all the expected components loaded', function() {
    browser.get('http://localhost:8000');
    expect(browser.getTitle()).toEqual('Kubernetes UI');

    // Navigate to the graph page.
    var dashboardTab = element(by.id('tab_001'));
    expect(dashboardTab).toBeDefined();
    dashboardTab.click();
    expect(browser.getLocationAbsUrl()).toBe('/dashboard/');

    // Verify if the views dropdown list has been loaded.
    var views = element(by.model('page'));
    expect(views).toBeDefined();
  });

  it('should have the subnav view', function() {
    browser.get('http://localhost:8000/');

    // Make sure the subnav and subpage selection is correctly loaded.
    expect(by.css('.dashboard-subnav')).toBeDefined();
    var selectSubPages = element(by.css('.selectSubPages'));
    expect(selectSubPages).toBeDefined();
    selectSubPages.click();

    var select = element(by.model('page'));
    expect(select).toBeDefined();

    // Make clicks to expand the subpage selection options.
    selectSubPages.click();
    select.click();

    expect(element(by.id('groupsView'))).toBeDefined();
    expect(element(by.id('podsView'))).toBeDefined();
    expect(element(by.id('minionsView'))).toBeDefined();
    expect(element(by.id('rcView'))).toBeDefined();
    expect(element(by.id('servicesView'))).toBeDefined();
    expect(element(by.id('eventsView'))).toBeDefined();
    expect(element(by.id('cAdvisorView'))).toBeDefined();
  });

  it('should have the cAdvisor view by default', function() {
    browser.get('http://localhost:8000/');
    expect(browser.getTitle()).toEqual('Kubernetes UI');

    expect(element.all(by.css('.dashboard')).count()).toBeGreaterThan(0);
    expect(element.all(by.css('.server-overview')).count()).toEqual(1);

    // Also we should render the view based on the minions data.
    expect(element(by.repeater("minion in minions.items"))).toBeDefined();

    // Make sure the svg is drawn.
    var svg = element(by.css('svg'));
    expect(svg).toBeDefined();
  });

  it('should have the correct subviews', function() {
    browser.get('http://localhost:8000/');

    var subviews = ['podsView', 'minionsView', 'rcView', 'servicesView', 'eventsView'];

    for (var i = 0; i < subviews.length; i++) {
      var subview = subviews[i];

      // Navigate to the subview.
      var select = element(by.model('page'));
      select.click();
      var podsView = element(by.id(subview));
      expect(podsView).toBeDefined();
      podsView.click();

      // Make sure the pods view still has the right title and subnav.
      expect(browser.getTitle()).toEqual('Kubernetes UI');
      expect(by.css('.dashboard-subnav')).toBeDefined();
      expect(element(by.css('.selectSubPages'))).toBeDefined();

      // Verify if the views dropdown list has been loaded.
      var views = element(by.model('page'));
      expect(views).toBeDefined();

      expect(element.all(by.css('.list-pods')).count()).toEqual(1);
      // Make sure we are populating the pods info dynamically.
      expect(element(by.repeater('h in headers'))).toBeDefined();
    }
  });

  it('should have the correct groups view', function() {
    browser.get('http://localhost:8000/');

    // Navigate to the group view.
    var select = element(by.model('page'));
    select.click();
    var groupsView = element(by.id('groupsView'));
    expect(groupsView).toBeDefined();
    groupsView.click();

    // Make sure the pods view still have the right title and subnav.
    expect(browser.getTitle()).toEqual('Kubernetes UI');
    expect(by.css('.dashboard-subnav')).toBeDefined();
    expect(element(by.css('.selectSubPages'))).toBeDefined();

    // Verify if the views dropdown list has been loaded.
    var views = element(by.model('page'));
    expect(views).toBeDefined();

    // Make sure necessary components are loaded correctly.
    var select = element(by.model('selectedGroupBy'));
    expect(select).toBeDefined();

    // Open the selection options.
    select.click();
    expect(element(by.repeater("g in groupByOptions"))).toBeDefined();
  });
});
