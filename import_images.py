import os
from icrawler.builtin import GoogleImageCrawler


main_folder = 'dataset'
os.makedirs(main_folder, exist_ok=True)


sports_celebrities = [
    "Lionel Messi", "Cristiano Ronaldo", "LeBron James", "Roger Federer", "Serena Williams",
    "Virat Kohli", "Usain Bolt", "Michael Jordan", "Tom Brady",
    "Sachin Tendulkar", "MS Dhoni", "Kapil Dev", "Saina Nehwal", "PV Sindhu",
    "Sunil Chhetri", "Mary Kom", "Milkha Singh", "Neeraj Chopra","Maria Sharapova",
    "Kobe Bryant", "Neymar Jr", "Rafael Nadal", "Novak Djokovic", "Tiger Woods",
     "Mike Tyson", "Shaquille O'Neal", "Dwayne Wade", "Stephen Curry"
]

max_images = 500

def download_images(celebrity_name, max_images=500):
    
    folder_name = os.path.join(main_folder, celebrity_name.replace(" ", "_"))
    os.makedirs(folder_name, exist_ok=True)
    
    google_crawler = GoogleImageCrawler(storage={"root_dir": folder_name})
    google_crawler.crawl(keyword=celebrity_name, max_num=max_images)


for celebrity in sports_celebrities:
    print(f"Downloading images for {celebrity}...")
    download_images(celebrity, max_images)
    print(f"Downloaded images for {celebrity}.")
