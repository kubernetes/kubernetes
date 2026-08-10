import { PROJECT_NUMBER } from './config';
import type { Github } from './github';

export const PRIORITIES = {
  p0: 'P0',
  p1: 'P1',
  p2: 'P2',
  p3: 'P3',
};

export const projectIds = async (github: Github) : Promise<{
  projectId: string;
  createFieldId: string;
  updateFieldId: string;
  authorFieldId: string;
  priorityField: {
    id: string;
    options: Record<keyof typeof PRIORITIES, string>;
  };
}> => {
  const projectInfo = await github.getProjectInfo(PROJECT_NUMBER);
  const projectId = projectInfo.data.repository.projectV2.id!;

  let createFieldId = undefined;
  let updateFieldId = undefined;
  let authorFieldId = undefined;
  let priorityField = {
    id: undefined,
    options: {
      p0: undefined,
      p1: undefined,
      p2: undefined,
      p3: undefined,
    },
  };

  for (const field of projectInfo.data.repository.projectV2.fields.nodes) {
    if (field.name === 'Create date') {
      createFieldId = field.id;
    }
    if (field.name === 'Update date') {
      updateFieldId = field.id;
    }
    if (field.name === 'Item author') {
      authorFieldId = field.id;
    }
    if (field.name === 'Priority') {
      priorityField.id = field.id;
      for (const option of field.options) {
        switch (option.name) {
          case 'P0':
            priorityField.options.p0 = option.id;
            break;
          case 'P1':
            priorityField.options.p1 = option.id;
            break;
          case 'P2':
            priorityField.options.p2 = option.id;
            break;
          case 'P3':
            priorityField.options.p3 = option.id;
        }
      }
    }
  }

  if (createFieldId === undefined) {
    throw new Error('Project field "Creation date" not found');
  }

  if (updateFieldId === undefined) {
    throw new Error('Project field "Update date" not found');
  }

  if (authorFieldId === undefined) {
    throw new Error('Project field "Item author" not found');
  }

  if (priorityField.id === undefined) {
    throw new Error('Project field "Priority" not found');
  }

  for (const [priorityLabel, priorityProjectLabel] of Object.entries(PRIORITIES)) {
    if (priorityField.options[priorityLabel as keyof typeof PRIORITIES] === undefined) {
      throw new Error(`Project field "Priority" does not have a "${priorityProjectLabel}" option.`);
    }
  }

  return {
    projectId,
    createFieldId,
    updateFieldId,
    authorFieldId,
    priorityField: priorityField as any,
  };
};

export const getPriorityFromLabels = (labels: any[], priorityField: { options: Record<keyof typeof PRIORITIES, string> }): string | undefined => {
  for (const label of labels) {
    const labelName = label.name.toLowerCase();
    if (labelName in PRIORITIES && priorityField.options[labelName as keyof typeof PRIORITIES]) {
      return priorityField.options[labelName as keyof typeof PRIORITIES];
    }
  }
  return undefined;
};

const botLogins = new Set([
  'github-actions',
  'maintainer-for-aws',
]);

export const isBotLogin = (login: string) => botLogins.has(login);
