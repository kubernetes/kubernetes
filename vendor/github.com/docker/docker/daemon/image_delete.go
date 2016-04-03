package daemon

import (
	"fmt"
	"strings"

	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/graph"
	"github.com/docker/docker/image"
	"github.com/docker/docker/pkg/parsers"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/docker/utils"
)

// FIXME: remove ImageDelete's dependency on Daemon, then move to graph/
func (daemon *Daemon) ImageDelete(name string, force, noprune bool) ([]types.ImageDelete, error) {
	list := []types.ImageDelete{}
	if err := daemon.imgDeleteHelper(name, &list, true, force, noprune); err != nil {
		return nil, err
	}
	if len(list) == 0 {
		return nil, fmt.Errorf("Conflict, %s wasn't deleted", name)
	}

	return list, nil
}

func (daemon *Daemon) imgDeleteHelper(name string, list *[]types.ImageDelete, first, force, noprune bool) error {
	var (
		repoName, tag string
		tags          = []string{}
	)
	repoAndTags := make(map[string][]string)

	// FIXME: please respect DRY and centralize repo+tag parsing in a single central place! -- shykes
	repoName, tag = parsers.ParseRepositoryTag(name)
	if tag == "" {
		tag = graph.DEFAULTTAG
	}

	if name == "" {
		return fmt.Errorf("Image name can not be blank")
	}

	img, err := daemon.Repositories().LookupImage(name)
	if err != nil {
		if r, _ := daemon.Repositories().Get(repoName); r != nil {
			return fmt.Errorf("No such image: %s", utils.ImageReference(repoName, tag))
		}
		return fmt.Errorf("No such image: %s", name)
	}

	if strings.Contains(img.ID, name) {
		repoName = ""
		tag = ""
	}

	byParents := daemon.Graph().ByParent()

	repos := daemon.Repositories().ByID()[img.ID]

	//If delete by id, see if the id belong only to one repository
	deleteByID := repoName == ""
	if deleteByID {
		for _, repoAndTag := range repos {
			parsedRepo, parsedTag := parsers.ParseRepositoryTag(repoAndTag)
			if repoName == "" || repoName == parsedRepo {
				repoName = parsedRepo
				if parsedTag != "" {
					repoAndTags[repoName] = append(repoAndTags[repoName], parsedTag)
				}
			} else if repoName != parsedRepo && !force && first {
				// the id belongs to multiple repos, like base:latest and user:test,
				// in that case return conflict
				return fmt.Errorf("Conflict, cannot delete image %s because it is tagged in multiple repositories, use -f to force", name)
			} else {
				//the id belongs to multiple repos, with -f just delete all
				repoName = parsedRepo
				if parsedTag != "" {
					repoAndTags[repoName] = append(repoAndTags[repoName], parsedTag)
				}
			}
		}
	} else {
		repoAndTags[repoName] = append(repoAndTags[repoName], tag)
	}

	if !first && len(repoAndTags) > 0 {
		return nil
	}

	if len(repos) <= 1 || (len(repoAndTags) <= 1 && deleteByID) {
		if err := daemon.canDeleteImage(img.ID, force); err != nil {
			return err
		}
	}

	// Untag the current image
	for repoName, tags := range repoAndTags {
		for _, tag := range tags {
			tagDeleted, err := daemon.Repositories().Delete(repoName, tag)
			if err != nil {
				return err
			}
			if tagDeleted {
				*list = append(*list, types.ImageDelete{
					Untagged: utils.ImageReference(repoName, tag),
				})
				daemon.EventsService.Log("untag", img.ID, "")
			}
		}
	}
	tags = daemon.Repositories().ByID()[img.ID]
	if (len(tags) <= 1 && repoName == "") || len(tags) == 0 {
		if len(byParents[img.ID]) == 0 {
			if err := daemon.Repositories().DeleteAll(img.ID); err != nil {
				return err
			}
			if err := daemon.Graph().Delete(img.ID); err != nil {
				return err
			}
			*list = append(*list, types.ImageDelete{
				Deleted: img.ID,
			})
			daemon.EventsService.Log("delete", img.ID, "")
			if img.Parent != "" && !noprune {
				err := daemon.imgDeleteHelper(img.Parent, list, false, force, noprune)
				if first {
					return err
				}

			}

		}
	}
	return nil
}

func (daemon *Daemon) canDeleteImage(imgID string, force bool) error {
	if daemon.Graph().IsHeld(imgID) {
		return fmt.Errorf("Conflict, cannot delete because %s is held by an ongoing pull or build", stringid.TruncateID(imgID))
	}
	for _, container := range daemon.List() {
		if container.ImageID == "" {
			// This technically should never happen, but if the container
			// has no ImageID then log the situation and move on.
			// If we allowed processing to continue then the code later
			// on would fail with a "Prefix can't be empty" error even
			// though the bad container has nothing to do with the image
			// we're trying to delete.
			logrus.Errorf("Container %q has no image associated with it!", container.ID)
			continue
		}
		parent, err := daemon.Repositories().LookupImage(container.ImageID)
		if err != nil {
			if daemon.Graph().IsNotExist(err, container.ImageID) {
				continue
			}
			return err
		}

		if err := daemon.graph.WalkHistory(parent, func(p image.Image) error {
			if imgID == p.ID {
				if container.IsRunning() {
					if force {
						return fmt.Errorf("Conflict, cannot force delete %s because the running container %s is using it, stop it and retry", stringid.TruncateID(imgID), stringid.TruncateID(container.ID))
					}
					return fmt.Errorf("Conflict, cannot delete %s because the running container %s is using it, stop it and use -f to force", stringid.TruncateID(imgID), stringid.TruncateID(container.ID))
				} else if !force {
					return fmt.Errorf("Conflict, cannot delete %s because the container %s is using it, use -f to force", stringid.TruncateID(imgID), stringid.TruncateID(container.ID))
				}
			}
			return nil
		}); err != nil {
			return err
		}
	}
	return nil
}
