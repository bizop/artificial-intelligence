// *********************************************
// CODE DEVELOPED BY: NATHAN WILBANKS
// FIND ME ON LINKEDIN: https://www.linkedin.com/in/nathanwilbanks/
// FIND ME ON TWITTER: https://twitter.com/NathanWilbanks_
// LICENSE: MIT License
// VERSION: 11.12.2023
// COPYRIGHT: 2023 Nathan Wilbanks

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// *********************************************

import fs from 'fs';
import path from 'path';
import axios from 'axios';
import FormData from 'form-data';

const imgurClientId = process.env.IMGUR_CLIENT_ID; // Replace with your Imgur Client ID
const inputFolder = './images'; // Folder with images
const outputFolder = './sorted_images'; // Where to store sorted images

async function uploadToImgur(filePath) {
  const image = fs.readFileSync(filePath);
  const formData = new FormData();
  formData.append('image', image);

  const response = await axios.post('https://api.imgur.com/3/image', formData, {
    headers: {
      Authorization: `Client-ID ${imgurClientId}`,
      ...formData.getHeaders(),
    },
  });

  return response.data.data.link;
}

async function labelImageWithHuggingFace(imageUrl) {
  const response = await axios.post(
    'https://api-inference.huggingface.co/models/hustvl/yolos-tiny',
    { url: imageUrl },
    {
      headers: {
        Authorization: `Bearer ${process.env.HUGGINGFACE_API_KEY}`,
      },
    }
  );
  return response.data;
}

async function categorizeWithOpenAI(labels) {
  const prompt = `Determine the best category for an image with these labels: ${labels.join(', ')}`;
  const response = await axios.post(
    'https://api.openai.com/v1/engines/davinci/completions',
    {
      prompt: prompt,
      max_tokens: 60,
    },
    {
      headers: {
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      },
    }
  );
  return response.data.choices[0].text.trim();
}

function createCategoryDirectory(category) {
  const categoryPath = path.join(outputFolder, category);
  if (!fs.existsSync(categoryPath)) {
    fs.mkdirSync(categoryPath, { recursive: true });
  }
  return categoryPath;
}

async function processImages() {
  const files = fs.readdirSync(inputFolder);

  for (const file of files) {
    const filePath = path.join(inputFolder, file);
    const imageUrl = await uploadToImgur(filePath);

    const huggingFaceResult = await labelImageWithHuggingFace(imageUrl);
    const labels = huggingFaceResult.map((item) => item.label);
    const category = await categorizeWithOpenAI(labels);
    const categoryPath = createCategoryDirectory(category);

    const newFilePath = path.join(categoryPath, file);
    fs.renameSync(filePath, newFilePath);
    console.log(`Moved ${file} to ${newFilePath}`);
  }
}

processImages().then(() => console.log('Image processing complete.'));
