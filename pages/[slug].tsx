import { getPostSlugs, getPostBySlug } from 'lib/api';
import { Post } from 'lib/models';
import { Params } from 'next/dist/server/router';
import React from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';

const PostPage: React.FC<Post> = ({ metadata, markdownBody }) => (
  <div>
    <h1>{metadata.title}</h1>
    <ReactMarkdown
      children={markdownBody}
      rehypePlugins={[rehypeKatex]}
      remarkPlugins={[remarkMath]}
    ></ReactMarkdown>
  </div>
);

export default PostPage;

export async function getStaticProps({ params }: Params) {
  const post = getPostBySlug(params.slug);
  return {
    props: { ...post },
  };
}

export async function getStaticPaths() {
  const slugs = getPostSlugs();
  return {
    paths: slugs.map((slug: string) => ({
      params: {
        slug,
      },
    })),
    fallback: false,
  };
}
