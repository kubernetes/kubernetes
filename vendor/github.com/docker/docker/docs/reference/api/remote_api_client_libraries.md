<!--[metadata]>
+++
title = "Remote API client libraries"
description = "Various client libraries available to use with the Docker remote API"
keywords = ["API, Docker, index, registry, REST, documentation, clients, C#, Erlang, Go, Groovy, Java, JavaScript, Perl, PHP, Python, Ruby, Rust,  Scala"]
[menu.main]
parent="mn_reference"
+++
<![end-metadata]-->

# Docker Remote API client libraries

These libraries have not been tested by the Docker maintainers for
compatibility. Please file issues with the library owners. If you find
more library implementations, please list them in Docker doc bugs and we
will add the libraries here.

<table border="1" class="docutils">
  <colgroup>
    <col width="24%">
    <col width="17%">
    <col width="48%">
    <col width="11%">
  </colgroup>
  <thead valign="bottom">
    <tr>
      <th class="head">Language/Framework</th>
      <th class="head">Name</th>
      <th class="head">Repository</th>
      <th class="head">Status</th>
    </tr>
  </thead>
  <tbody valign = "top">
    <tr>
      <td>C#</td>
      <td>Docker.DotNet</td>
      <td><a class="reference external" href="https://github.com/ahmetalpbalkan/Docker.DotNet">https://github.com/ahmetalpbalkan/Docker.DotNet</a></td>
      <td>Active</td>
    </tr>
    <tr>
      <td>C++</td>
      <td>lasote/docker_client</td>
      <td><a class="reference external" href="http://www.biicode.com/lasote/docker_client">http://www.biicode.com/lasote/docker_client (Biicode C++ dependency manager)</a></td>
      <td>Active</td>
    </tr>
    <tr>
      <td>Erlang</td>
      <td>erldocker</td>
      <td><a class="reference external" href="https://github.com/proger/erldocker">https://github.com/proger/erldocker</a></td>
      <td>Active</td>
    </tr>
    <tr>
      <td>Dart</td>
      <td>bwu_docker</td>
      <td><a class="reference external" href="https://github.com/bwu-dart/bwu_docker">https://github.com/bwu-dart/bwu_docker</a></td>
      <td>Active</td>
    </tr>
    <tr>
      <td>Go</td>
      <td>go-dockerclient</td>
      <td><a class="reference external" href="https://github.com/fsouza/go-dockerclient">https://github.com/fsouza/go-dockerclient</a></td>
      <td>Active</td>
    </tr>
    <tr>
      <td>Go</td>
      <td>dockerclient</td>
      <td><a class="reference external" href="https://github.com/samalba/dockerclient">https://github.com/samalba/dockerclient</a></td>
      <td>Active</td>
    </tr>
    <tr>
      <td>Gradle</td>
      <td>gradle-docker-plugin</td>
      <td><a class="reference external" href="https://github.com/gesellix/gradle-docker-plugin">https://github.com/gesellix/gradle-docker-plugin</a></td>
      <td>Active</td>
    </tr>
    <tr>
      <td>Groovy</td>
      <td>docker-client</td>
      <td><a class="reference external" href="https://github.com/gesellix/docker-client">https://github.com/gesellix/docker-client</a></td>
      <td>Active</td>
    </tr>
    <tr>
      <td>Haskell</td>
      <td>docker-hs</td>
      <td><a class="reference external" href="https://github.com/denibertovic/docker-hs">https://github.com/denibertovic/docker-hs</a></td>
      <td>Active</td>
    </tr>
    <tr>
      <td>Java</td>
      <td>docker-java</td>
      <td><a class="reference external" href="https://github.com/docker-java/docker-java">https://github.com/docker-java/docker-java</a></td>
      <td>Active</td>
    </tr>
    <tr>
      <td>Java</td>
      <td>docker-client</td>
      <td><a class="reference external" href="https://github.com/spotify/docker-client">https://github.com/spotify/docker-client</a></td>
      <td>Active</td>
    </tr>
    <tr>
      <td>Java</td>
      <td>jclouds-docker</td>
      <td><a class="reference external" href="https://github.com/jclouds/jclouds-labs/tree/master/docker">https://github.com/jclouds/jclouds-labs/tree/master/docker</a></td>
      <td>Active</td>
    </tr>
    <tr>
      <td>JavaScript (NodeJS)</td>
      <td>dockerode</td>
      <td><a class="reference external" href="https://github.com/apocas/dockerode">https://github.com/apocas/dockerode</a>
  Install via NPM: <cite>npm install dockerode</cite></td>
      <td>Active</td>
    </tr>
    <tr>
      <td>JavaScript (NodeJS)</td>
      <td>docker.io</td>
      <td><a class="reference external" href="https://github.com/appersonlabs/docker.io">https://github.com/appersonlabs/docker.io</a>
  Install via NPM: <cite>npm install docker.io</cite></td>
      <td>Active</td>
    </tr>
    <tr>
      <td>JavaScript</td>
      <td>docker-js</td>
      <td><a class="reference external" href="https://github.com/dgoujard/docker-js">https://github.com/dgoujard/docker-js</a></td>
      <td>Outdated</td>
    </tr>
    <tr>
      <td>JavaScript (Angular) <strong>WebUI</strong></td>
      <td>docker-cp</td>
      <td><a class="reference external" href="https://github.com/13W/docker-cp">https://github.com/13W/docker-cp</a></td>
      <td>Active</td>
    </tr>
    <tr>
      <td>JavaScript (Angular) <strong>WebUI</strong></td>
      <td>dockerui</td>
      <td><a class="reference external" href="https://github.com/crosbymichael/dockerui">https://github.com/crosbymichael/dockerui</a></td>
      <td>Active</td>
    </tr>
    <tr>
      <td>JavaScript (Angular) <strong>WebUI</strong></td>
      <td>dockery</td>
      <td><a class="reference external" href="https://github.com/lexandro/dockery">https://github.com/lexandro/dockery</a></td>
      <td>Active</td>
    </tr>    
    <tr>
      <td>Perl</td>
      <td>Net::Docker</td>
      <td><a class="reference external" href="https://metacpan.org/pod/Net::Docker">https://metacpan.org/pod/Net::Docker</a></td>
      <td>Active</td>
    </tr>
    <tr>
      <td>Perl</td>
      <td>Eixo::Docker</td>
      <td><a class="reference external" href="https://github.com/alambike/eixo-docker">https://github.com/alambike/eixo-docker</a></td>
      <td>Active</td>
    </tr>
    <tr>
      <td>PHP</td>
      <td>Alvine</td>
      <td><a class="reference external" href="http://pear.alvine.io/">http://pear.alvine.io/</a> (alpha)</td>
      <td>Active</td>
    </tr>
    <tr>
      <td>PHP</td>
      <td>Docker-PHP</td>
      <td><a class="reference external" href="http://stage1.github.io/docker-php/">http://stage1.github.io/docker-php/</a></td>
      <td>Active</td>
    </tr>
    <tr>
      <td>Python</td>
      <td>docker-py</td>
      <td><a class="reference external" href="https://github.com/docker/docker-py">https://github.com/docker/docker-py</a></td>
      <td>Active</td>
    </tr>
    <tr>
      <td>Ruby</td>
      <td>docker-api</td>
      <td><a class="reference external" href="https://github.com/swipely/docker-api">https://github.com/swipely/docker-api</a></td>
      <td>Active</td>
    </tr>
    <tr>
      <td>Ruby</td>
      <td>docker-client</td>
      <td><a class="reference external" href="https://github.com/geku/docker-client">https://github.com/geku/docker-client</a></td>
      <td>Outdated</td>
    </tr>
    <tr>
      <td>Rust</td>
      <td>docker-rust</td>
      <td><a class="reference external" href="https://github.com/abh1nav/docker-rust">https://github.com/abh1nav/docker-rust</a></td>
      <td>Active</td>
    </tr>
    <tr>
      <td>Scala</td>
      <td>tugboat</td>
      <td><a class="reference external" href="https://github.com/softprops/tugboat">https://github.com/softprops/tugboat</a></td>
      <td>Active</td>
    </tr>
    <tr>
      <td>Scala</td>
      <td>reactive-docker</td>
      <td><a class="reference external" href="https://github.com/almoehi/reactive-docker">https://github.com/almoehi/reactive-docker</a></td>
      <td>Active</td>
    </tr>
  </tbody>
</table>

