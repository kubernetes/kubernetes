/**
 Copyright 2015 Google Inc. All rights reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

describe('Kubernetes UI Graph', function() {
  it('should have all the expected components loaded', function() {
    browser.get('http://localhost:8000');
    expect(browser.getTitle()).toEqual('Kubernetes UI');

    // Navigate to the graph page.
    var graphTab = element(by.id('tab_002'));
    expect(graphTab).toBeDefined();
    graphTab.click();
    expect(browser.getLocationAbsUrl()).toBe('/graph/');

    // Verify if the control action icons have been loaded.
    var expandCollapse = element(by.id('ExpandCollapse'));
    expect(expandCollapse).toBeDefined();
    var toggleSelect = element(by.id('ToggleSelect'));
    expect(toggleSelect).toBeDefined();
    var toggleSource = element(by.id('ToggleSource'));
    expect(toggleSource).toBeDefined();
    var pollOnce = element(by.id('PollOnce'));
    expect(pollOnce).toBeDefined();

    // Use mock data to ease testing.
    toggleSource.click();
    // Just pull once to get a stable graph.
    pollOnce.click();

    // Make sure the svg is drawn.
    var svg = element(by.css('svg'));
    expect(svg).toBeDefined();
  });

  it('should have all the details pane working correctly', function() {
    browser.get('http://localhost:8000');
    expect(browser.getTitle()).toEqual('Kubernetes UI');

    // Navigate to the graph page.
    var graphTab = element(by.id('tab_002'));
    expect(graphTab).toBeDefined();
    graphTab.click();
    expect(browser.getLocationAbsUrl()).toBe('/graph/');

    var toggleBtn = element(by.id('toggleDetails'));
    expect(toggleBtn).toBeDefined();
    expect(element(by.repeater('type in getLegendLinkTypes()'))).toBeDefined();

    var details = element(by.id('details'));
    expect(details).toBeDefined();
    expect(details.isDisplayed()).toBe(false);

    toggleBtn.click();
    expect(details.isDisplayed()).toBe(true);
  });

  it('should have all the graph working correctly', function() {
    browser.get('http://localhost:8000');
    expect(browser.getTitle()).toEqual('Kubernetes UI');

    // Navigate to the graph page.
    var graphTab = element(by.id('tab_002'));
    expect(graphTab).toBeDefined();
    graphTab.click();
    expect(browser.getLocationAbsUrl()).toBe('/graph/');

    var svg = element(by.css('d3-visualization svg'));
    expect(svg).toBeDefined();

    // Make sure the graph is drawn with necessary components.
    expect(element(by.css('d3-visualization svg marker'))).toBeDefined();
    expect(element(by.css('d3-visualization svg text'))).toBeDefined();
    expect(element(by.css('d3-visualization svg path'))).toBeDefined();
    expect(element(by.css('d3-visualization svg image'))).toBeDefined();
    expect(element(by.css('d3-visualization svg circle'))).toBeDefined();

    var toggleSource = element(by.id('ToggleSource'));
    expect(toggleSource).toBeDefined();
    var pollOnce = element(by.id('PollOnce'));
    expect(pollOnce).toBeDefined();

    // Use mock data to ease testing.
    toggleSource.click();
    // Just pull once to get a stable graph.
    pollOnce.click();

    // Add a custom locator to match on the d3 data backing the circle element.
    by.addLocator('datumIdMatches', function(datumId, opt_parentElement, opt_rootSelector) {
      var matchingCircles = [];

      window.d3.selectAll('circle').each(function(d) {
        if (d && d.id === datumId) {
          matchingCircles.push(this);
        }
      });

      return matchingCircles;
    });

    // This id matches a node defined in /www/master/shared/assets/sampleData1.json.
    var firstNode = element(by.datumIdMatches('Pod:redis-slave-controller-vi7hv'));
    expect(firstNode).toBeDefined();

    // Now click to select this node.
    firstNode.click();
    // Make sure the details pane should be showing something.
    var details = element(by.id('details'));
    expect(details).toBeDefined();
    expect(details.isDisplayed()).toBe(true);
    // Also make sure the details are populated from real data.
    expect(element(by.repeater('(tag, value) in getSelectionDetails()'))).toBeDefined();

    // Now ensure we can navigate to the node inspection page.
    var inspectBtn = element(by.id('inspectBtn'));
    expect(inspectBtn).toBeDefined();
    inspectBtn.click();
    // Check if we arrive at the inspection page.
    expect(browser.getLocationAbsUrl()).toBe('/graph/inspect');

    // Ensure the inspection page has the details populated.
    expect(element(by.repeater('(key, val) in json track by $index'))).toBeDefined();

    // Ensure the inspection page has a back button and it works.
    var backBtn = element(by.id('backButton'));
    expect(backBtn).toBeDefined();
    backBtn.click();
    expect(browser.getLocationAbsUrl()).toBe('/graph/');
  });

});
