import { GetStaticPropsResult } from 'next';
import Link from 'next/link';
import React from 'react';

import { getAllPostTitles } from '../lib/api';
import { Metadata } from '../lib/models';

const IndexPage: React.FC<{ metadataList: Metadata[] }> = ({
  metadataList,
}) => {
  return (
    <ul>
      {metadataList.map((metadata) => (
        <li key={metadata.slug}>
          <Link href={metadata.slug}>
            <a>{metadata.title}</a>
          </Link>
        </li>
      ))}
    </ul>
  );
};

export default IndexPage;

export async function getStaticProps(): Promise<
  GetStaticPropsResult<{ metadataList: Metadata[] }>
> {
  const metadataList = await getAllPostTitles();
  return {
    props: {
      metadataList,
    },
  };
}
