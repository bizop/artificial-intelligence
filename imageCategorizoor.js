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
