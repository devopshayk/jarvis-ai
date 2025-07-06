import os
import requests
from PIL import Image
from io import BytesIO
import re
import shutil
import glob
import random

class ImageSearcher:
    def __init__(self, search_term=None):
        self.search_term = search_term
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Referer': 'https://duckduckgo.com/',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        self.ensure_searched_images_folder()

    def ensure_searched_images_folder(self):
        if not os.path.exists('searched_images'):
            os.makedirs('searched_images')

    def get_vqd(self, query):
        url = 'https://duckduckgo.com/'
        params = {'q': query}
        res = self.session.get(url, params=params)
        if res.status_code == 200:
            match = re.search(r'vqd=([\d-]+)&', res.text)
            if match:
                return match.group(1)
        return None

    def is_specific_query(self, query):
        # If the query contains numbers, or symbols like Ω, μ, or a mix of letters and numbers, treat as specific
        # Contains any digit or common electronic symbols
        if re.search(r'[0-9ΩμµkKMR]', query):
            return True
        # Contains a mix of letters and numbers (e.g., 1N4007, 104, A1B2)
        if re.search(r'[A-Za-z]+[0-9]+|[0-9]+[A-Za-z]+', query):
            return True
        return False

    def fetch_image_duckduckgo(self, query):
        if self.is_specific_query(query):
            print("[INFO] Specific query detected. Accepting first available image.")
            vqd = self.get_vqd(query)
            if not vqd:
                print("Could not get vqd token")
                return None
            url = 'https://duckduckgo.com/i.js'
            params = {'q': query, 'vqd': vqd, 'o': 'json'}
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                results = response.json()
                if results['results']:
                    for i, result in enumerate(results['results'][:10]):
                        image_url = result['image']
                        print(f"DuckDuckGo Image URL [{i+1}]: {image_url}")
                        try:
                            img_response = self.session.get(image_url, headers={
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
                                'Referer': 'https://duckduckgo.com/',
                            })
                            img_response.raise_for_status()
                            img_data = img_response.content
                            img = Image.open(BytesIO(img_data))
                            print(f"[SUCCESS] Found image at result {i+1}")
                            return img
                        except Exception as e:
                            print(f"[ERROR] Failed to fetch or open image [{i+1}]: {e}")
                    print("[ERROR] None of the first 10 images could be fetched or opened.")
                    return None
                else:
                    print("[ERROR] No images from DuckDuckGo. Full response:")
                    print(results)
                    return None
            except Exception as e:
                print(f"[ERROR] DuckDuckGo error: {e}")
                return None
        else:
            # Enhance the query to prefer transparent PNGs and stickers
            enhanced_query = f"{query} png transparent no background sticker cutout"
            print(f"[INFO] General query detected. Trying for transparent PNGs: {enhanced_query}")
            vqd = self.get_vqd(enhanced_query)
            if not vqd:
                print("Could not get vqd token")
                return None
            url = 'https://duckduckgo.com/i.js'
            params = {'q': enhanced_query, 'vqd': vqd, 'o': 'json'}
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                results = response.json()
                if results['results']:
                    # Try the first 20 image results, prefer PNGs with transparency
                    for i, result in enumerate(results['results'][:20]):
                        image_url = result['image']
                        print(f"DuckDuckGo Image URL [{i+1}]: {image_url}")
                        if not image_url.lower().endswith('.png'):
                            continue  # Only try PNGs
                        try:
                            img_response = self.session.get(image_url, headers={
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
                                'Referer': 'https://duckduckgo.com/',
                            })
                            img_response.raise_for_status()
                            img_data = img_response.content
                            img = Image.open(BytesIO(img_data))
                            # Check for alpha channel (transparency)
                            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                                print(f"[SUCCESS] Found transparent PNG at result {i+1}")
                                return img
                            else:
                                print(f"[INFO] PNG at result {i+1} is not transparent.")
                        except Exception as e:
                            print(f"[ERROR] Failed to fetch or open image [{i+1}]: {e}")
                    print("[ERROR] No transparent PNG found in the first 20 results.")
                    return None
                else:
                    print("[ERROR] No images from DuckDuckGo. Full response:")
                    print(results)
                    return None
            except Exception as e:
                print(f"[ERROR] DuckDuckGo error: {e}")
                return None

    def save_image(self, img, query):
        # Always save to the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        safe_query = query.strip().replace(' ', '_').replace('/', '_').lower()
        
        # Try to save to both searched_images and images directories
        searched_path = os.path.join(script_dir, 'searched_images', f'{safe_query}.png')
        images_path = os.path.join(script_dir, 'images', f'{safe_query}.png')
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(searched_path), exist_ok=True)
        os.makedirs(os.path.dirname(images_path), exist_ok=True)
        
        print(f"[DEBUG] Current working directory: {os.getcwd()}")
        print(f"[DEBUG] Attempting to save image to: {os.path.abspath(searched_path)}")
        
        try:
            # Save to searched_images
            img.save(searched_path, format='PNG')
            print(f"Image saved as {searched_path}!")
            
            # Also save to images directory for compatibility
            img.save(images_path, format='PNG')
            print(f"Image also saved as {images_path}!")
            
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save image: {e}")
            return False

    def get_component_folder_name(self, query):
        synonyms = {
            'condensator': 'capacitor',
            'electrolyticcondensator': 'electrolyticcapacitor',
            'electrolyticcapacitor': 'electrolyticcapacitor',
            'led': 'lightemittingdiode',
            'ic': 'integratedcircuit',
            'opamp': 'operationalamplifier',
            'transistor': 'transistor',
            'resistor': 'resistor',
            'capacitor': 'capacitor',
            'diode': 'diode',
            'potentiometer': 'potentiometer',
            'inductor': 'inductor',
            'crystaloscillator': 'crystaloscillator',
            'switch': 'switch',
        }
        norm = query.strip().lower().replace(' ', '')
        return synonyms.get(norm, norm) or ""

    def find_component_folder(self, dataset_dir, folder_name):
        # Look for a folder matching the name (case-insensitive, singular/plural)
        candidates = [folder_name, folder_name + 's', folder_name + 'es']
        folders = os.listdir(dataset_dir)
        for candidate in candidates:
            for f in folders:
                if f.lower() == candidate.lower():
                    return os.path.join(dataset_dir, f)
        return None

    def get_local_component_image(self, query):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(script_dir, 'archive', 'dataset')
        folder_name = self.get_component_folder_name(query)
        if not folder_name:
            return None
        folder_path = self.find_component_folder(dataset_dir, folder_name)
        if folder_path and os.path.isdir(folder_path):
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                chosen_file = random.choice(image_files)
                img_path = os.path.join(folder_path, chosen_file)
                print(f"[INFO] Using local image: {img_path}")
                try:
                    img = Image.open(img_path)
                    return img
                except Exception as e:
                    print(f"[ERROR] Failed to open local image: {e}")
        return None

    def copy_local_component_image(self, query):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(script_dir, 'archive', 'dataset')
        folder_name = self.get_component_folder_name(query)
        if not folder_name:
            return False
        folder_path = self.find_component_folder(dataset_dir, folder_name)
        if folder_path and os.path.isdir(folder_path):
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                chosen_file = random.choice(image_files)
                src_path = os.path.join(folder_path, chosen_file)
                dest_folder = os.path.join(script_dir, 'searched_images')
                if not os.path.exists(dest_folder):
                    os.makedirs(dest_folder)
                dest_path = os.path.join(dest_folder, chosen_file)
                print(f"[INFO] Copying local image: {src_path} to {dest_path}")
                try:
                    shutil.copy2(src_path, dest_path)
                    print(f"Image copied to {dest_path}")
                    return True
                except Exception as e:
                    print(f"[ERROR] Failed to copy local image: {e}")
        return False

    def run(self):
        if not self.search_term:
            print("No search term set!")
            return
        print(f"Searching for: {self.search_term}")
        normalized = self.search_term.strip().replace(' ', '').lower()
        if normalized in ['electronicboard', 'eb']:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            eb_folder = os.path.join(script_dir, 'Electronic Board')
            # Search for any common image extension
            possible_exts = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
            found = False
            for ext in possible_exts:
                eb_path = os.path.join(eb_folder, f'electronic_board{ext}')
                if os.path.exists(eb_path):
                    dest_folder = os.path.join(script_dir, 'searched_images')
                    if not os.path.exists(dest_folder):
                        os.makedirs(dest_folder)
                    dest_path = os.path.join(dest_folder, f'electronic_board{ext}')
                    print(f"[DEBUG] Copying {eb_path} to {dest_path}")
                    shutil.copy2(eb_path, dest_path)
                    print(f"Image copied to {dest_path}")
                    found = True
                    break
            if not found:
                print(f"[ERROR] No electronic_board image found in {eb_folder} with extensions {possible_exts}")
            return
        # Special handling for electrolytic components
        if any(x in normalized for x in ['electrolytic', 'electrolyticcapacitor']):
            if self.copy_local_component_image('electrolyticcapacitor'):
                print("Search term has been reset to blank (local electrolytic component image).")
                self.search_term = ""
                return
        # Check for local component image (general query)
        if not self.is_specific_query(self.search_term):
            local_img = self.get_local_component_image(self.search_term)
            if local_img:
                if self.save_image(local_img, self.search_term):
                    self.search_term = ""
                    print("Search term has been reset to blank (local image).")
                else:
                    print("[ERROR] Failed to save local image. Search term remains unchanged.")
                return
        # Otherwise, use web search as before
        img = self.fetch_image_duckduckgo(self.search_term)
        if img:
            if self.save_image(img, self.search_term):
                self.search_term = ""
                print("Search term has been reset to blank.")
            else:
                print("[ERROR] Failed to save image. Search term remains unchanged.")
        else:
            print("[ERROR] No image found or failed to fetch image. Search term remains unchanged.")

if __name__ == "__main__":
    searcher = ImageSearcher("resistor")  # Change the search term here
    searcher.run()
    print(f"After run, search_term is: '{searcher.search_term}'")