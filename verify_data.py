"""
ตรวจสอบข้อมูล Netflix CSV และแสดงรายละเอียด
"""
import pandas as pd
from pathlib import Path

# โหลดข้อมูล
data_path = Path("data/netflix_titles.csv")

if not data_path.exists():
    print(f"❌ ไม่พบไฟล์: {data_path}")
    exit(1)

print("🔍 กำลังตรวจสอบข้อมูล Netflix CSV...")
print("=" * 60)

# อ่านไฟล์
df = pd.read_csv(data_path)

# ข้อมูลพื้นฐาน
print(f"\n📊 ข้อมูลทั่วไป:")
print(f"   จำนวนแถว: {len(df):,}")
print(f"   จำนวน Columns: {len(df.columns)}")

# รายชื่อ Columns
print(f"\n📝 Columns ทั้งหมด:")
for i, col in enumerate(df.columns, 1):
    dtype = df[col].dtype
    non_null = df[col].notna().sum()
    null_count = df[col].isna().sum()
    print(f"   {i}. {col:<20} | Type: {str(dtype):<10} | Non-null: {non_null:>5} | Null: {null_count:>5}")

# ตัวอย่างข้อมูล 5 แถวแรก
print(f"\n📄 ตัวอย่างข้อมูล 5 แถวแรก:")
print(df.head().to_string())

# สถิติตาม Type
print(f"\n📺 ประเภทเนื้อหา:")
if 'type' in df.columns:
    type_counts = df['type'].value_counts()
    for content_type, count in type_counts.items():
        print(f"   {content_type}: {count:,} ({count/len(df)*100:.1f}%)")

# ช่วงปี
print(f"\n📅 ช่วงปีที่เผยแพร่:")
if 'release_year' in df.columns:
    print(f"   ปีแรกสุด: {df['release_year'].min()}")
    print(f"   ปีล่าสุด: {df['release_year'].max()}")
    print(f"   ค่าเฉลี่ย: {df['release_year'].mean():.0f}")

# Rating
print(f"\n⭐ Rating ทั้งหมด:")
if 'rating' in df.columns:
    rating_counts = df['rating'].value_counts().head(10)
    for rating, count in rating_counts.items():
        print(f"   {rating}: {count:,}")

# Genres
print(f"\n🎭 Top 10 Genres:")
if 'listed_in' in df.columns:
    genres = df['listed_in'].dropna().str.split(',').explode().str.strip()
    top_genres = genres.value_counts().head(10)
    for genre, count in top_genres.items():
        print(f"   {genre}: {count:,}")

# Countries
print(f"\n🌍 Top 10 ประเทศ:")
if 'country' in df.columns:
    countries = df['country'].dropna().str.split(',').explode().str.strip()
    top_countries = countries.value_counts().head(10)
    for country, count in top_countries.items():
        if country:  # ตรวจสอบไม่ให้เป็นค่าว่าง
            print(f"   {country}: {count:,}")

# ตรวจสอบข้อมูลที่หายไป
print(f"\n⚠️ ข้อมูลที่หายไป (มากกว่า 5%):")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
for col in df.columns:
    if missing_pct[col] > 5:
        print(f"   {col}: {missing[col]:,} ({missing_pct[col]:.1f}%)")

# ตรวจสอบ duplicates
print(f"\n🔄 ข้อมูลซ้ำ:")
duplicates = df.duplicated(subset=['title']).sum()
print(f"   ชื่อซ้ำ: {duplicates:,} รายการ")

# ตัวอย่างชื่อเรื่อง
print(f"\n🎬 ตัวอย่างชื่อเรื่อง 10 รายการแรก:")
for i, title in enumerate(df['title'].head(10), 1):
    print(f"   {i}. {title}")

print("\n" + "=" * 60)
print("✅ ตรวจสอบเสร็จสิ้น!")
print("\nข้อมูลพร้อมใช้งาน ✓")