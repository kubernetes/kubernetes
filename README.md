Kubernetes.io Documentation
=

####Author: [Tim Garrison](https://github.com/mitnosirrag) \<timg@plexipixel.com\>
####Updated: `November 17, 2014`

Overview
-
Kubernetes.io is a public facing website for Kubernetes by Google.  It is intended to provide documentation, support and information for developers looking to learn more about Kubernetes.  The website is a GitHub.io page and part of the Google Cloud Platform account at <https://github.com/googlecloudplatform/>.  Kubernetes.io is a static website that utilizes the `Jekyll` framework in order to make edits and updates easier to manage.

GitHub.io Pages
-
See: [GitHub.io Pages](https://pages.github.com/)

Technologies Used
-
- [Jekyll](http://jekyllrb.com/docs/home/) - Framework used to process dynamic files into static website. Automatically run when `push`ing to GitHub. 
- [Liquid Templates](https://github.com/Shopify/liquid/wiki) - Templating system, similar to Handlebars or Django, used for processing views with minimal logic. Embeds into HTML, similar to PHP.
- [SASS](http://sass-lang.com/documentation/file.SASS_REFERENCE.html) - Pre-processor for CSS. Allows for re-usable items when styling.
- JavaScript / [jQuery](http://api.jquery.com/) - jQuery is a JavaScript framework that makes common JS tasks a lot easier.
- [Markdown](http://daringfireball.net/projects/markdown/syntax) - Markdown can be used to create HTML without knowing HTML, but you have to know Markdown. This document is created in Markdown. Kubernetes.io doesn't really use Markdown, since it wasn't practical.
- [YAML](http://www.yaml.org/) - Data serialization format that is human-readable. Used for storing all variables and their values. Essentially our model.

Folder Structure
-
- **`/`**
    - **`_config.yml`** - Site configuration. Meta values, navigation, Jekyll settings and global variables are stored here. Changes to this file can be catastrophic.
    - **`_local_config.yml`** - Used for local testing. Anything set here will override values set in *`_config.yml`*.
    - **`community.md`** - Variables and content that are displayed in the *`community.html`* layout. Variables are stored in YAML format.
    - **`events.md`** - Variables and content that are displayed in the *events.html* layout. Variables are stored in YAML format.
    - **`feed.xml`** - Not currently used
    - **`gettingstarted.md`** - Variables and content that are displayed in the *`gettingstarted.html`* layout. Variables are stored in YAML format.
    - **`index.html`** - Variables and content that are displayed in the *`homepage.html`* layout. Variables are stored in YAML format.
    - **`news.md`** - Variables and content that are stored in the *news.html* layout. Variables are stored in YAML format.
- **`/_data`** - YAML files that populate dynamic sections with content.
    - **`/events.yml`** - A YAML file describing events. Events follow a simple YAML template structure that uses key/value pairs to set variables that are displayed.  Events are sorted in the order they appear in the file. New events should be added to the top of the file to maintain a reverse chronological sort.
    - **/`news`** - Contains YAML files for individual news stories.  News files follow a simple YAML template structure that uses key/value pairs to set variables that are displayed.  News, like events, are sorted reverse chronologically, and must be named with their date in `YYYY-MM-DD.yml` format.  News files with duplicate dates that conflict with another file should have a single digit appended to the end of the filename to allow proper sorting.
- **`/_includes`** - Generic HTML/Liquid includes for all pages.
    - **`head.html`** - HTML declaration, meta tags, style tags and script tags. This file is included in all layouts.
    - **`header.html`** - Included in the *`head.html`* include, this file displays the navigation for both desktop and mobile.
    - **`hero.html`** - Included in the *`header.html`* include, this file displays the hero section of each interior page and the home page.
    - **`footer.html`** - This file in included in all layouts.  It displays all footer information.
- **`/_layouts`** - HTML/Liquid markup templates.
    - **`community.html`** - HTML markup for community page.
    - **`default.html`** - All layouts inherit this layout to simplify includes. 
    - **`events.html`** - HTML markup for events page.
    - **`gettingstarted.html`** - HTML markup for getting started page.
    - **`homepage.html`** - HTML markup for home page.
    - **`news.html`** - HTML markup for news page.
- **`/_posts`** - Not currently used, would be for blogging.
- **`/_sass`** - SASS partials are stored here. Jekyll will automatically parse SASS files into CSS files.  If you add a new partial file, make sure to include it in *`/css/main.scss`*.
- **`/_site`** - Jekyll will generate the static website into this directory. Anything manually placed here will be overridden.
- **`/css`** - Generic CSS can be placed here.  The file *`main.scss`* is what generates SASS partials, alter it with extreme caution.
- **`/img`** - Site images and SVGs are stored here, in appropriate sub folders.
- **`/js`** - Javascript and jQuery files are stored here.
- **`/legacy`** - Contains the old version of the kubernetes.io website. Files in here can be linked to until they are replaced with newer versions.

Events file example
-
#### Filename: `events.yml`
```yaml
- 
    date: 2014-11-22 00:00:00 -0800
    title: "Top100 Summit"
    location: "Beijing, China"
    venue: "Beijing International Convention Center"
    speaker: "Tim Hockin & Dawn Chen"
    url: "http://www.top100summit.com/"
- 
    date: 2014-11-05 00:00:00 -0800
    title: "AppSphere 2014"
    location: "Las Vegas, NV, USA"
    venue: "Containers in the Cloud: AppDynamics and Kubernetes"
    speaker: "Eric Johnson"
    url: "https://appsphere2014.secure.mfactormeetings.com/agenda/"
```
##### Note that dates follow the format: `YYYY-MM-DD HH:II:SS +0000`

News file example
-
#### Filename: `2014-11-04.yml`
```yaml
date: 2014-11-04 00:00:00 -0800
headline: "Google unveils its Container Engine to run apps in the best possible way on its cloud"
author: "Jordan Novet"
publication: "VentureBeat"
url: "http://venturebeat.com/2014/11/04/google-container-engine/"
```
##### Note that dates follow the format: `YYYY-MM-DD HH:II:SS +0000`
