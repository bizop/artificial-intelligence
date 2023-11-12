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

const huggingFaceAPI = process.env.HUGGINGFACE_API_KEY; // Replace with your HuggingFace API key
const openaiAPI = process.env.OPENAI_API_KEY; // Replace with your OpenAI API key
const inputFolder = './images'; // Folder with images
const outputFolder = './sorted_images'; // Where to store sorted images

// Optional function if your image recognition model needs a URL
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

async function labelImageWithHuggingFace(filePath) {
  const data = fs.readFileSync(filePath);
  const response = await axios.post('https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large', data, {
    headers: {
      Authorization: `Bearer ${huggingFaceAPI}`,
      'Content-Type': 'application/octet-stream',
    },
  });
  return response.data;
}

async function categorizeWithOpenAI(labels) {
  const prompt = `Determine the best category for an image with these labels (People, Houses, Motorcycles, etc). 
  Do not acknowledge this request. Only return a single category name, nothing else. 
  Labels: ${labels}`;

  const messages = [
    {
      role: 'system',
      content: ``,
    },
    {
      role: 'user',
      content: prompt,
    },
  ];

  const response = await axios({
    method: 'post',
    url: 'https://api.openai.com/v1/chat/completions',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${openaiAPI}`,
    },
    data: {
      model: 'gpt-3.5-turbo-1106',
      messages: messages,
      max_tokens: 60,
      temperature: 1,
    },
  });

  const finalCompleteText = response.data.choices[0].message.content;
  return finalCompleteText.trim();
}

function sanitizeDirectoryName(name) {
  return name.replace(/[^a-zA-Z0-9-_]/g, '_');
}

function createCategoryDirectory(category) {
  const sanitizedCategory = sanitizeDirectoryName(category);
  const categoryPath = path.join(outputFolder, sanitizedCategory);
  if (!fs.existsSync(categoryPath)) {
    fs.mkdirSync(categoryPath, { recursive: true });
  }
  return categoryPath;
}

async function processImages() {
  const files = fs.readdirSync(inputFolder);

  for (const file of files) {
    const filePath = path.join(inputFolder, file);

    const huggingFaceResult = await labelImageWithHuggingFace(filePath);
    const labels = huggingFaceResult[0].generated_text;
    const category = await categorizeWithOpenAI(labels);
    const categoryPath = createCategoryDirectory(category);

    const newFilePath = path.join(categoryPath, file);
    fs.copyFileSync(filePath, newFilePath); // Copy file to new directory
    console.log(`Copied ${file} to ${newFilePath}`);
  }
}

processImages().then(() => console.log('Image processing complete.'));
