package graph

import (
	"errors"
	"fmt"
	"net"
	"net/url"
	"strings"
	"time"

	"github.com/Sirupsen/logrus"
	"github.com/docker/distribution/registry/client/transport"
	"github.com/docker/docker/image"
	"github.com/docker/docker/pkg/progressreader"
	"github.com/docker/docker/pkg/streamformatter"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/docker/registry"
	"github.com/docker/docker/utils"
)

type v1Puller struct {
	*TagStore
	endpoint registry.APIEndpoint
	config   *ImagePullConfig
	sf       *streamformatter.StreamFormatter
	repoInfo *registry.RepositoryInfo
	session  *registry.Session
}

func (p *v1Puller) Pull(tag string) (fallback bool, err error) {
	if utils.DigestReference(tag) {
		// Allowing fallback, because HTTPS v1 is before HTTP v2
		return true, registry.ErrNoSupport{errors.New("Cannot pull by digest with v1 registry")}
	}

	tlsConfig, err := p.registryService.TlsConfig(p.repoInfo.Index.Name)
	if err != nil {
		return false, err
	}
	// Adds Docker-specific headers as well as user-specified headers (metaHeaders)
	tr := transport.NewTransport(
		// TODO(tiborvass): was ReceiveTimeout
		registry.NewTransport(tlsConfig),
		registry.DockerHeaders(p.config.MetaHeaders)...,
	)
	client := registry.HTTPClient(tr)
	v1Endpoint, err := p.endpoint.ToV1Endpoint(p.config.MetaHeaders)
	if err != nil {
		logrus.Debugf("Could not get v1 endpoint: %v", err)
		return true, err
	}
	p.session, err = registry.NewSession(client, p.config.AuthConfig, v1Endpoint)
	if err != nil {
		// TODO(dmcgowan): Check if should fallback
		logrus.Debugf("Fallback from error: %s", err)
		return true, err
	}
	if err := p.pullRepository(tag); err != nil {
		// TODO(dmcgowan): Check if should fallback
		return false, err
	}
	return false, nil
}

func (p *v1Puller) pullRepository(askedTag string) error {
	out := p.config.OutStream
	out.Write(p.sf.FormatStatus("", "Pulling repository %s", p.repoInfo.CanonicalName))

	repoData, err := p.session.GetRepositoryData(p.repoInfo.RemoteName)
	if err != nil {
		if strings.Contains(err.Error(), "HTTP code: 404") {
			return fmt.Errorf("Error: image %s not found", utils.ImageReference(p.repoInfo.RemoteName, askedTag))
		}
		// Unexpected HTTP error
		return err
	}

	logrus.Debugf("Retrieving the tag list")
	tagsList := make(map[string]string)
	if askedTag == "" {
		tagsList, err = p.session.GetRemoteTags(repoData.Endpoints, p.repoInfo.RemoteName)
	} else {
		var tagId string
		tagId, err = p.session.GetRemoteTag(repoData.Endpoints, p.repoInfo.RemoteName, askedTag)
		tagsList[askedTag] = tagId
	}
	if err != nil {
		if err == registry.ErrRepoNotFound && askedTag != "" {
			return fmt.Errorf("Tag %s not found in repository %s", askedTag, p.repoInfo.CanonicalName)
		}
		logrus.Errorf("unable to get remote tags: %s", err)
		return err
	}

	for tag, id := range tagsList {
		repoData.ImgList[id] = &registry.ImgData{
			ID:       id,
			Tag:      tag,
			Checksum: "",
		}
	}

	logrus.Debugf("Registering tags")
	// If no tag has been specified, pull them all
	if askedTag == "" {
		for tag, id := range tagsList {
			repoData.ImgList[id].Tag = tag
		}
	} else {
		// Otherwise, check that the tag exists and use only that one
		id, exists := tagsList[askedTag]
		if !exists {
			return fmt.Errorf("Tag %s not found in repository %s", askedTag, p.repoInfo.CanonicalName)
		}
		repoData.ImgList[id].Tag = askedTag
	}

	errors := make(chan error)

	layersDownloaded := false
	imgIDs := []string{}
	sessionID := p.session.ID()
	defer func() {
		p.graph.Release(sessionID, imgIDs...)
	}()
	for _, image := range repoData.ImgList {
		downloadImage := func(img *registry.ImgData) {
			if askedTag != "" && img.Tag != askedTag {
				errors <- nil
				return
			}

			if img.Tag == "" {
				logrus.Debugf("Image (id: %s) present in this repository but untagged, skipping", img.ID)
				errors <- nil
				return
			}

			// ensure no two downloads of the same image happen at the same time
			if c, err := p.poolAdd("pull", "img:"+img.ID); err != nil {
				if c != nil {
					out.Write(p.sf.FormatProgress(stringid.TruncateID(img.ID), "Layer already being pulled by another client. Waiting.", nil))
					<-c
					out.Write(p.sf.FormatProgress(stringid.TruncateID(img.ID), "Download complete", nil))
				} else {
					logrus.Debugf("Image (id: %s) pull is already running, skipping: %v", img.ID, err)
				}
				errors <- nil
				return
			}
			defer p.poolRemove("pull", "img:"+img.ID)

			// we need to retain it until tagging
			p.graph.Retain(sessionID, img.ID)
			imgIDs = append(imgIDs, img.ID)

			out.Write(p.sf.FormatProgress(stringid.TruncateID(img.ID), fmt.Sprintf("Pulling image (%s) from %s", img.Tag, p.repoInfo.CanonicalName), nil))
			success := false
			var lastErr, err error
			var isDownloaded bool
			for _, ep := range p.repoInfo.Index.Mirrors {
				ep += "v1/"
				out.Write(p.sf.FormatProgress(stringid.TruncateID(img.ID), fmt.Sprintf("Pulling image (%s) from %s, mirror: %s", img.Tag, p.repoInfo.CanonicalName, ep), nil))
				if isDownloaded, err = p.pullImage(img.ID, ep, repoData.Tokens); err != nil {
					// Don't report errors when pulling from mirrors.
					logrus.Debugf("Error pulling image (%s) from %s, mirror: %s, %s", img.Tag, p.repoInfo.CanonicalName, ep, err)
					continue
				}
				layersDownloaded = layersDownloaded || isDownloaded
				success = true
				break
			}
			if !success {
				for _, ep := range repoData.Endpoints {
					out.Write(p.sf.FormatProgress(stringid.TruncateID(img.ID), fmt.Sprintf("Pulling image (%s) from %s, endpoint: %s", img.Tag, p.repoInfo.CanonicalName, ep), nil))
					if isDownloaded, err = p.pullImage(img.ID, ep, repoData.Tokens); err != nil {
						// It's not ideal that only the last error is returned, it would be better to concatenate the errors.
						// As the error is also given to the output stream the user will see the error.
						lastErr = err
						out.Write(p.sf.FormatProgress(stringid.TruncateID(img.ID), fmt.Sprintf("Error pulling image (%s) from %s, endpoint: %s, %s", img.Tag, p.repoInfo.CanonicalName, ep, err), nil))
						continue
					}
					layersDownloaded = layersDownloaded || isDownloaded
					success = true
					break
				}
			}
			if !success {
				err := fmt.Errorf("Error pulling image (%s) from %s, %v", img.Tag, p.repoInfo.CanonicalName, lastErr)
				out.Write(p.sf.FormatProgress(stringid.TruncateID(img.ID), err.Error(), nil))
				errors <- err
				return
			}
			out.Write(p.sf.FormatProgress(stringid.TruncateID(img.ID), "Download complete", nil))

			errors <- nil
		}

		go downloadImage(image)
	}

	var lastError error
	for i := 0; i < len(repoData.ImgList); i++ {
		if err := <-errors; err != nil {
			lastError = err
		}
	}
	if lastError != nil {
		return lastError
	}

	for tag, id := range tagsList {
		if askedTag != "" && tag != askedTag {
			continue
		}
		if err := p.Tag(p.repoInfo.LocalName, tag, id, true); err != nil {
			return err
		}
	}

	requestedTag := p.repoInfo.LocalName
	if len(askedTag) > 0 {
		requestedTag = utils.ImageReference(p.repoInfo.LocalName, askedTag)
	}
	WriteStatus(requestedTag, out, p.sf, layersDownloaded)
	return nil
}

func (p *v1Puller) pullImage(imgID, endpoint string, token []string) (bool, error) {
	history, err := p.session.GetRemoteHistory(imgID, endpoint)
	if err != nil {
		return false, err
	}
	out := p.config.OutStream
	out.Write(p.sf.FormatProgress(stringid.TruncateID(imgID), "Pulling dependent layers", nil))
	// FIXME: Try to stream the images?
	// FIXME: Launch the getRemoteImage() in goroutines

	sessionID := p.session.ID()
	// As imgID has been retained in pullRepository, no need to retain again
	p.graph.Retain(sessionID, history[1:]...)
	defer p.graph.Release(sessionID, history[1:]...)

	layersDownloaded := false
	for i := len(history) - 1; i >= 0; i-- {
		id := history[i]

		// ensure no two downloads of the same layer happen at the same time
		if c, err := p.poolAdd("pull", "layer:"+id); err != nil {
			logrus.Debugf("Image (id: %s) pull is already running, skipping: %v", id, err)
			<-c
		}
		defer p.poolRemove("pull", "layer:"+id)

		if !p.graph.Exists(id) {
			out.Write(p.sf.FormatProgress(stringid.TruncateID(id), "Pulling metadata", nil))
			var (
				imgJSON []byte
				imgSize int
				err     error
				img     *image.Image
			)
			retries := 5
			for j := 1; j <= retries; j++ {
				imgJSON, imgSize, err = p.session.GetRemoteImageJSON(id, endpoint)
				if err != nil && j == retries {
					out.Write(p.sf.FormatProgress(stringid.TruncateID(id), "Error pulling dependent layers", nil))
					return layersDownloaded, err
				} else if err != nil {
					time.Sleep(time.Duration(j) * 500 * time.Millisecond)
					continue
				}
				img, err = image.NewImgJSON(imgJSON)
				layersDownloaded = true
				if err != nil && j == retries {
					out.Write(p.sf.FormatProgress(stringid.TruncateID(id), "Error pulling dependent layers", nil))
					return layersDownloaded, fmt.Errorf("Failed to parse json: %s", err)
				} else if err != nil {
					time.Sleep(time.Duration(j) * 500 * time.Millisecond)
					continue
				} else {
					break
				}
			}

			for j := 1; j <= retries; j++ {
				// Get the layer
				status := "Pulling fs layer"
				if j > 1 {
					status = fmt.Sprintf("Pulling fs layer [retries: %d]", j)
				}
				out.Write(p.sf.FormatProgress(stringid.TruncateID(id), status, nil))
				layer, err := p.session.GetRemoteImageLayer(img.ID, endpoint, int64(imgSize))
				if uerr, ok := err.(*url.Error); ok {
					err = uerr.Err
				}
				if terr, ok := err.(net.Error); ok && terr.Timeout() && j < retries {
					time.Sleep(time.Duration(j) * 500 * time.Millisecond)
					continue
				} else if err != nil {
					out.Write(p.sf.FormatProgress(stringid.TruncateID(id), "Error pulling dependent layers", nil))
					return layersDownloaded, err
				}
				layersDownloaded = true
				defer layer.Close()

				err = p.graph.Register(img,
					progressreader.New(progressreader.Config{
						In:        layer,
						Out:       out,
						Formatter: p.sf,
						Size:      imgSize,
						NewLines:  false,
						ID:        stringid.TruncateID(id),
						Action:    "Downloading",
					}))
				if terr, ok := err.(net.Error); ok && terr.Timeout() && j < retries {
					time.Sleep(time.Duration(j) * 500 * time.Millisecond)
					continue
				} else if err != nil {
					out.Write(p.sf.FormatProgress(stringid.TruncateID(id), "Error downloading dependent layers", nil))
					return layersDownloaded, err
				} else {
					break
				}
			}
		}
		out.Write(p.sf.FormatProgress(stringid.TruncateID(id), "Download complete", nil))
	}
	return layersDownloaded, nil
}
