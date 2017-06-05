<!--[metadata]>
+++
draft = true
+++
<![end-metadata]-->

# Glossary

This page contains definitions for distribution related terms.

<dl>
	<dt id="blob"><h4>Blob</h4></dt>
	<dd>
      <blockquote>A blob is any kind of content that is stored by a Registry under a content-addressable identifier (a "digest").</blockquote>
      <p>
      	<a href="#layer">Layers</a> are a good example of "blobs".
      </p>
	</dd>

	<dt id="image"><h4>Image</h4></dt>
	<dd>
      <blockquote>An image is a named set of immutable data from which a Docker container can be created.</blockquote>
      <p>
      An image is represented by a json file called a <a href="#manifest">manifest</a>, and is conceptually a set of <a hred="#layer">layers</a>.

      Image names indicate the location where they can be pulled from and pushed to, as they usually start with a <a href="#registry">registry</a> domain name and port.

      </p>
    </dd>

	<dt id="layer"><h4>Layer</h4></dt>
	<dd>
	<blockquote>A layer is a tar archive bundling partial content from a filesystem.</blockquote>
	<p>
	Layers from an <a href="#image">image</a> are usually extracted in order on top of each other to make up a root filesystem from which containers run out.
	</p>
	</dd>

	<dt id="manifest"><h4>Manifest</h4></dt>
	<dd><blockquote>A manifest is the JSON representation of an image.</blockquote></dd>

	<dt id="namespace"><h4>Namespace</h4></dt>
	<dd><blockquote>A namespace is a collection of repositories with a common name prefix.</blockquote>
	<p>
	The namespace with an empty prefix is considered the Global Namespace.
	</p>
	</dd>

	<dt id="registry"><h4>Registry</h4></dt>
	<dd><blockquote>A registry is a service that let you store and deliver <a href="#images">images</a>.</blockquote>
	</dd>

	<dt id="registry"><h4>Repository</h4></dt>
	<dd>
	<blockquote>A repository is a set of data containing all versions of a given image.</blockquote>
	</dd>

	<dt id="scope"><h4>Scope</h4></dt>
	<dd><blockquote>A scope is the portion of a namespace onto which a given authorization token is granted.</blockquote></dd>

	<dt id="tag"><h4>Tag</h4></dt>
	<dd><blockquote>A tag is conceptually a "version" of a <a href="#image">named image</a>.</blockquote>
	<p>
	Example: `docker pull myimage:latest` instructs docker to pull the image "myimage" in version "latest".
	</p>
	
	</dd>
	

</dl>
