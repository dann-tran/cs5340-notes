import fs from 'fs';
import matter from 'gray-matter';
import { join } from 'path';

import { Metadata, Post } from './models';

const postsDirectory = join(process.cwd(), '_posts');

export function getPostSlugs(): string[] {
  return fs
    .readdirSync(postsDirectory)
    .map((filename) => filename.replace(/\.md$/, ''));
}

export function getPostBySlug(slug: string): Post {
  const fullPath = join(postsDirectory, `${slug}.md`);
  const fileContents = fs.readFileSync(fullPath, 'utf8');
  const { data, content } = matter(fileContents);
  return {
    metadata: {
      slug,
      title: data.title,
      tags: data.tags,
      lectureNumber: parseInt(data.lectureNumber),
    },
    markdownBody: content,
  };
}

export function getAllPosts(): Post[] {
  return getPostSlugs()
    .map(getPostBySlug)
    .sort((a, b) => a.metadata.lectureNumber - b.metadata.lectureNumber);
}

export function getAllPostMetadata(): Metadata[] {
  return getAllPosts().map((post) => post.metadata);
}
